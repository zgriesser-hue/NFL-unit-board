"""Tune and upgrade the NBA total model, then test new information against the closing totals.

Step 1  TUNE the rating settings (how much to trust last season, how fast old games fade, how much rating carries over, ridge strength)
        by walk-forward error on the actual totals of seasons that have NO closing lines in our file, so the seasons used to grade the
        model against the market never influence the settings.
Step 2  GRADE the old and the tuned model on the seasons with closing lines (vs the market).
Step 3  NEW INFORMATION: add travel and venue features (miles since the last game, body-clock shift, road-trip length, altitude) and
        a referee-crew scoring tendency, and see whether each one shrinks the gap to the market. Referees need data/nba_refs.csv
        (from nba_fetch_refs.py); without it that part is skipped.
Run:  python nba_total_tune.py     (needs data/nba_games.csv and data/nba_lines_hist.csv)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import nba_model_pipeline as P

log = P.log
BASE = {"K_PRIOR": P.K_PRIOR, "DECAY": P.DECAY, "CARRY": P.CARRY, "ALPHA": P.ALPHA}
GRID = {"K_PRIOR": [4, 8, 12, 20], "DECAY": [0.85, 0.88, 0.91, 0.94, 0.96], "CARRY": [0.0, 0.15, 0.3, 0.4, 0.55], "ALPHA": [3, 10, 30]}
REF_K = 100.0          # games' worth of weight pulling a referee's tendency toward average


def set_params(**kw):
    for k, v in kw.items():
        setattr(P, k, v)


def build(g, params):
    set_params(**params)
    d, _ = P.history_table(g)
    return d


def add_refs(d, refs):
    """ref_tot: the crew's average scoring tendency (points above the league level, shrunk), using only games BEFORE each date."""
    d = d.merge(refs, on="game_id", how="left").sort_values(["date", "game_id"]).reset_index(drop=True)
    st, out = {}, np.zeros(len(d))
    resid = (d["total"] - P.lg_total(d)).values
    cols = d[["ref1", "ref2", "ref3"]].values
    for _, idx in d.groupby("date", sort=True).indices.items():
        for i in idx:
            v = [st.get(r, (0.0, 0.0)) for r in cols[i] if isinstance(r, str)]
            out[i] = np.mean([s / (n + REF_K) for n, s in v]) if v else 0.0
        for i in idx:
            for r in cols[i]:
                if isinstance(r, str):
                    n, s = st.get(r, (0.0, 0.0))
                    st[r] = (n + 1.0, s + resid[i])
    d["ref_tot"] = out
    d["has_refs"] = d["ref1"].notna()
    return d


def walk(d, seasons, kind="one", cols=None):
    """Walk-forward predictions of the total for the given seasons; trained only on earlier seasons (skipping the very first)."""
    out = []
    first = d["season"].min()
    for s in seasons:
        tr, te = (d["season"] < s) & (d["season"] > first), d["season"] == s
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        t = d[te][["game_id", "season", "date", "home", "away", "total"]].copy()
        use = P.FEATS_T + list(cols or [])
        keep = P.FEATS + [c for c in use if c not in P.FEATS]
        t["p"] = P.TotalRidge(use).fit(d.loc[tr, keep], d.loc[tr, "total"]).predict(d.loc[te, keep]).values
        out.append(t)
    return pd.concat(out) if out else None


def mae(r):
    return float((r["p"] - r["total"]).abs().mean())


def tune(g, seasons_val):
    cur, best = dict(BASE), None
    t0 = time.time()
    for k in GRID:
        scores = {}
        for v in GRID[k]:
            scores[v] = mae(walk(build(g, {**cur, k: v}), seasons_val))
        v = min(scores, key=scores.get)
        cur[k], best = v, scores[v]
        edge = " (EDGE OF THE RANGE TRIED)" if v in (GRID[k][0], GRID[k][-1]) else ""
        log(f"  {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in scores.items()) + f"  -> {v}{edge}   ({time.time() - t0:.0f}s)")
    return cur, best


def grade(r, h):
    m = r.merge(h, on=["date", "home", "away"], how="inner")
    e = m["p"] - m["mkt"]
    return m, {"games": len(m), "model_miss": mae(m), "market_miss": float((m["mkt"] - m["total"]).abs().mean()),
               "corr_vs_market_miss": float(np.corrcoef(e, m["total"] - m["mkt"])[0, 1])}


def main():
    gp, hp = f"{P.DATA}/nba_games.csv", f"{P.DATA}/nba_lines_hist.csv"
    if not (os.path.exists(gp) and os.path.exists(hp)):
        log("Need data/nba_games.csv and data/nba_lines_hist.csv.")
        return 1
    g = P.load_games(gp)
    d0 = build(g, BASE)
    seasons = sorted(d0["season"].unique())
    have = P.seasons_with_lines(d0) or set()
    test = [s for s in seasons if s in have and (d0["season"] == s).sum() >= 1100 and s >= seasons[0] + 2][-P.N_TEST:]
    val = [s for s in seasons if s not in have and s >= seasons[0] + 2 and (d0["season"] == s).sum() >= 1100]
    log(f"{len(d0)} games; tuning on seasons without closing lines {[int(x) for x in val]}; grading against the market on {[int(x) for x in test]}")
    if not val or not test:
        log("Not enough seasons for a clean tune/test split.")
        return 1

    log("\nSTEP 1: TUNING (mean miss on the actual total over the tuning seasons, lower is better)")
    log(f"  starting point {BASE}: {mae(walk(d0, val)):.3f}")
    best, s_best = tune(g, val)
    log(f"  tuned: {s_best:.3f} with {best}")

    h = pd.read_csv(hp)
    h["date"] = pd.to_datetime(h["date"])
    h = h[["date", "home", "away", "total"]].rename(columns={"total": "mkt"})
    log("\nSTEP 2: GRADE against the closing totals (these seasons were never used for tuning)")
    _, a = grade(walk(d0, test), h)
    d1 = build(g, best)
    _, b = grade(walk(d1, test), h)
    rows = [{"model": "old settings", **a}, {"model": "tuned settings", **b}]

    log("\nSTEP 3: NEW INFORMATION on top of the tuned settings")
    refs = None
    rp = f"{P.DATA}/nba_refs.csv"
    if os.path.exists(rp):
        refs = pd.read_csv(rp, dtype={"game_id": str})
        refs = refs[refs["ref1"].notna()]
    dd = d1
    variants = {"+ travel and venue": P.TRAVEL}
    if refs is not None and len(refs) > 500:
        dd = add_refs(d1, refs[["game_id", "ref1", "ref2", "ref3"]])
        cov = dd[dd["season"].isin(test)]["has_refs"].mean()
        log(f"  referee data covers {dd['has_refs'].mean():.0%} of all games and {cov:.0%} of the graded games")
        variants["+ referee crew"] = ["ref_tot"]
        variants["+ travel, venue and referees"] = P.TRAVEL + ["ref_tot"]
    else:
        log("  (no referee data yet: run nba_fetch_refs.py until it says complete, then re-run this)")
    m1, _ = grade(walk(dd, test), h)
    for name, cols in variants.items():
        r = walk(dd, test, cols=cols)
        m, s = grade(r, h)
        rows.append({"model": name, **s})
        if "referee" in name and refs is not None:
            m = m.merge(dd[["game_id", "has_refs"]], on="game_id")
            log(f"    ({name}: {int(m['has_refs'].sum())} of the graded games have a referee crew; the rest get a neutral value)")
        variants[name] = (cols, r)
    res = pd.DataFrame(rows)
    res["gap"] = res["model_miss"] - res["market_miss"]
    log(res[["model", "games", "model_miss", "market_miss", "gap", "corr_vs_market_miss"]].round(3).to_string(index=False))

    # what the new features are worth, in plain numbers
    allc = P.FEATS_T + P.TRAVEL + (["ref_tot"] if "ref_tot" in dd else [])
    tr = dd["season"] > dd["season"].min()
    mdl = P.TotalRidge(allc).fit(dd.loc[tr, P.FEATS + [c for c in allc if c not in P.FEATS]], dd.loc[tr, "total"])
    sd = dd.loc[tr, allc].std()
    log("\nWEIGHTS in a model with everything (points of total per 1 standard deviation of the feature; trained on all seasons):")
    log(mdl.coef.reindex(P.TRAVEL + (["ref_tot"] if "ref_tot" in dd else [])).round(2).to_string())

    best_name = min(rows, key=lambda x: x["model_miss"])["model"]
    if best_name in variants and isinstance(variants[best_name], tuple):
        _, r = variants[best_name]
    else:
        r = walk(d1, test)
    m, _ = grade(r, h)
    log(f"\nBest: {best_name}. Win rate betting its side against the closing total (need > 52.4%):")
    log(P.win_record(m["p"] - m["mkt"], m["total"] - m["mkt"], P.TOTAL_EDGES).round(3).to_string(index=False))
    log(f"\nSettings to use: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
