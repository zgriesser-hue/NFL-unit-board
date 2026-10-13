"""Tune and upgrade the NBA total model, then test it against the closing totals.

Step 1  TUNE the rating settings (how much to trust last season, how fast old games fade, how much rating carries over, ridge strength)
        by walk-forward error on the actual totals of seasons that have NO closing lines in our file, so the seasons used to grade the
        model against the market never influence the settings.
Step 2  TWO-PART MODEL: a total is possessions x points per possession. Predict each part, then multiply.
Step 3  GRADE the old model, the tuned model, the two-part model and a blend on the seasons with closing lines (vs the market).
Run:  python nba_total_tune.py     (needs data/nba_games.csv and data/nba_lines_hist.csv)
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import nba_model_pipeline as P

log = P.log
BASE = {"K_PRIOR": P.K_PRIOR, "DECAY": P.DECAY, "CARRY": P.CARRY, "ALPHA": P.ALPHA}
GRID = {"K_PRIOR": [4, 8, 12, 20, 30], "DECAY": [0.94, 0.96, 0.97, 0.98, 0.99], "CARRY": [0.4, 0.55, 0.7, 0.85], "ALPHA": [10, 30, 100, 300]}
POSS_COLS = ["pace_h", "pace_a", "rest_h", "rest_a", "b2b_h", "b2b_a"]
EFF_COLS = ["off_h", "def_h", "off_a", "def_a", "rest_h", "rest_a", "b2b_h", "b2b_a"]


class TwoPart:
    """total = possessions x (points per 100 possessions summed over both teams) / 100, each part predicted relative to the league."""

    def fit(self, X, d):
        self.mp = P.Ridged().fit(X[POSS_COLS], d["poss_raw"] - X["lg_pace"])
        self.me = P.Ridged().fit(X[EFF_COLS], d["ortg_sum"] - 2 * X["lg_ortg"])
        return self

    def predict(self, X):
        poss = self.mp.predict(X[POSS_COLS]) + X["lg_pace"]
        eff = self.me.predict(X[EFF_COLS]) + 2 * X["lg_ortg"]
        return poss * eff / 100.0


def set_params(**kw):
    for k, v in kw.items():
        setattr(P, k, v)


def build(g, params):
    set_params(**params)
    d, _ = P.history_table(g)
    return d


def walk(d, seasons, kind):
    """Walk-forward predictions of the total for the given seasons; trained only on earlier seasons (skipping the very first)."""
    out = []
    first = d["season"].min()
    for s in seasons:
        tr, te = (d["season"] < s) & (d["season"] > first), d["season"] == s
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        t = d[te][["game_id", "season", "date", "home", "away", "total"]].copy()
        if kind == "one":
            t["p"] = P.TotalRidge().fit(d.loc[tr, P.FEATS], d.loc[tr, "total"]).predict(d.loc[te, P.FEATS]).values
        else:
            t["p"] = TwoPart().fit(d.loc[tr], d.loc[tr]).predict(d.loc[te]).values
        out.append(t)
    return pd.concat(out) if out else None


def mae(r):
    return float((r["p"] - r["total"]).abs().mean())


def tune(g, seasons_val, kind):
    cur, best = dict(BASE), None
    t0 = time.time()
    for rnd in (1,):                                     # one pass; a second pass changed nothing in testing
        for k in GRID:
            scores = {}
            for v in GRID[k]:
                p = {**cur, k: v}
                r = walk(build(g, p), seasons_val, kind)
                scores[v] = mae(r)
            v = min(scores, key=scores.get)
            cur[k] = v
            best = scores[v]
            log(f"  [{kind}] round {rnd} {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in scores.items()) + f"  -> {v}   ({time.time() - t0:.0f}s)")
    return cur, best


def main():
    gp, hp = f"{P.DATA}/nba_games.csv", f"{P.DATA}/nba_lines_hist.csv"
    if not (os.path.exists(gp) and os.path.exists(hp)):
        log("Need data/nba_games.csv and data/nba_lines_hist.csv.")
        return 1
    g = P.load_games(gp)
    d0 = build(g, BASE)
    seasons = sorted(d0["season"].unique())
    have = P.seasons_with_lines(d0) or set()
    test_all = [s for s in seasons if s in have and (d0["season"] == s).sum() >= 1100]
    test = [s for s in test_all if s >= seasons[0] + 2][-P.N_TEST:]
    val = [s for s in seasons if s not in have and s >= seasons[0] + 2 and (d0["season"] == s).sum() >= 1100]
    log(f"{len(d0)} games; tuning on seasons without closing lines {[int(x) for x in val]}; grading against the market on {[int(x) for x in test]}")
    if not val or not test:
        log("Not enough seasons for a clean tune/test split.")
        return 1

    log("\nSTEP 1/2: TUNING (mean miss on the actual total over the tuning seasons, lower is better)")
    base_one = mae(walk(d0, val, "one"))
    base_two = mae(walk(d0, val, "two"))
    log(f"  starting point (K_PRIOR={BASE['K_PRIOR']}, DECAY={BASE['DECAY']}, CARRY={BASE['CARRY']}, ALPHA={BASE['ALPHA']}): one-part {base_one:.3f}, two-part {base_two:.3f}")
    p_one, s_one = tune(g, val, "one")
    p_two, s_two = tune(g, val, "two")
    log(f"\n  tuned one-part {s_one:.3f} with {p_one}")
    log(f"  tuned two-part {s_two:.3f} with {p_two}")

    log("\nSTEP 3: GRADE against the closing totals (these seasons were never used for tuning)")
    h = pd.read_csv(hp)
    h["date"] = pd.to_datetime(h["date"])
    h = h[["date", "home", "away", "total"]].rename(columns={"total": "mkt"})
    runs = {"old model (as before)": (BASE, "one"), "tuned one-part": (p_one, "one"), "tuned two-part": (p_two, "two")}
    preds, rows = {}, []
    for name, (p, kind) in runs.items():
        r = walk(build(g, p), test, kind)
        preds[name] = r.merge(h, on=["date", "home", "away"], how="inner")
    names = list(runs)
    a, b = preds["tuned one-part"], preds["tuned two-part"]
    bl = a.merge(b[["game_id", "p"]], on="game_id", suffixes=("", "_2"))
    bl["p"] = (bl["p"] + bl["p_2"]) / 2
    preds["blend of tuned one + two"] = bl
    for name, r in preds.items():
        e = r["p"] - r["mkt"]
        rows.append({"model": name, "games": len(r), "model_miss": mae(r), "market_miss": float((r["mkt"] - r["total"]).abs().mean()),
                     "gap": mae(r) - float((r["mkt"] - r["total"]).abs().mean()),
                     "corr_vs_market_miss": float(np.corrcoef(e, r["total"] - r["mkt"])[0, 1])})
    log(pd.DataFrame(rows).round(3).to_string(index=False))
    best = min(rows, key=lambda x: x["model_miss"])["model"]
    r = preds[best]
    log(f"\nBest: {best}. Win rate betting its side against the closing total (need > 52.4%):")
    log(P.win_record(r["p"] - r["mkt"], r["total"] - r["mkt"], P.TOTAL_EDGES).round(3).to_string(index=False))
    log("\nSettings to use:  one-part " + str(p_one) + "   two-part " + str(p_two))
    return 0


if __name__ == "__main__":
    sys.exit(main())
