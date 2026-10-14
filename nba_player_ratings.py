"""Do better player ratings improve the NBA models?  (second ceiling test: the model is told who actually played)

Player ratings come from a ridge regression on past games ("game-level adjusted plus-minus"): for every team-game, the team's points per
100 possessions (relative to the league) is regressed on the minutes share of each of its own players (their offense) and of each opposing
player (their defense, i.e. how much scoring they take away). Ridge keeps players with little data near zero. Ratings are refit at the start
of each month using only earlier games, with older games counting less.
A game's features: the strength of the lineup that played (expected minutes x rating), both as a level and as a deviation from the team's
usual lineup. Compared with the crude game-score availability from the first test, on the same seasons, against the closing lines.
Needs data/nba_games.csv, data/nba_players.parquet (nba_fetch_players.py) and data/nba_lines_hist.csv.
Run:  python nba_player_ratings.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

import nba_model_pipeline as P

log = P.log
HALF_DAYS = float(os.environ.get("NBA_PR_HALF_DAYS", "500"))      # how fast older games lose weight in the player-rating fit
ALPHAS = [30.0, 100.0, 300.0, 1000.0]
NEWF = ["po_h", "pd_h", "po_a", "pd_a"]                            # lineup offense / defense (level)
DEVF = ["dpo_h", "dpd_h", "dpo_a", "dpd_a"]                        # the same, minus the team's usual


def prep(g, d, players):
    """Index players, build the sparse design (one row per team-game) and the per-game lineups."""
    pl = players[(players["min"] > 0) & players["team"].isin(set(P.NAMES.values()))].copy()
    pl["pid"] = pl["pid"].astype(str)
    pl["gid"] = pl["gid"].astype(str)
    pids = {p: i for i, p in enumerate(pl["pid"].unique())}
    pl["ix"] = pl["pid"].map(pids)
    lineup = {}                                                    # (gid, team) -> (ix array, minutes array)
    for (gid, team), x in pl.groupby(["gid", "team"], sort=False):
        lineup[(gid, team)] = (x["ix"].values, x["min"].values)
    gg = g.merge(d[["game_id", "lg_ortg"]], on="game_id", how="inner")
    n, rows, cols, vals, y, dates, gids = len(pids), [], [], [], [], [], []
    r = 0
    for x in gg.itertuples():
        a, b = lineup.get((x.game_id, x.home)), lineup.get((x.game_id, x.away))
        if a is None or b is None:
            continue
        for own, opp, flag, ortg in ((a, b, 1.0, x.h_ortg), (b, a, -1.0, x.a_ortg)):
            rows += [r] * (len(own[0]) + len(opp[0]) + 1)
            cols += list(own[0]) + [n + i for i in opp[0]] + [2 * n]
            vals += list(own[1] / 48.0) + list(opp[1] / 48.0) + [flag]
            y.append(ortg - x.lg_ortg)
            dates.append(x.date)
            r += 1
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(r, 2 * n + 1))
    return pl, pids, lineup, X, np.array(y), pd.to_datetime(pd.Series(dates)).values


def lineup_features(d, pl, pids, lineup, X, y, dates, alpha):
    """Walk through the games in date order; refit the ratings at the start of each month (earlier games only)."""
    n = len(pids)
    coef_o, coef_d = np.zeros(n), np.zeros(n)
    last_fit, st, usual = None, {}, {}
    out = {}
    d = d.sort_values(["date", "game_id"])
    for date, day in d.groupby("date", sort=True):
        key = (date.year, date.month)
        if key != last_fit:
            last_fit = key
            m = dates < np.datetime64(date)
            if m.sum() > 1500:
                w = 0.5 ** ((np.datetime64(date) - dates[m]).astype("timedelta64[D]").astype(float) / HALF_DAYS)
                mdl = Ridge(alpha=alpha, solver="sparse_cg", max_iter=300).fit(X[m], y[m], sample_weight=w)
                coef_o, coef_d = mdl.coef_[:n], mdl.coef_[n:2 * n]
        todo = []
        for gm in day.itertuples():
            res = {}
            for side, team in (("h", gm.home), ("a", gm.away)):
                lu = lineup.get((gm.game_id, team))
                if lu is None:
                    res[side] = (0.0, 0.0, 0.0, 0.0, team)
                    continue
                ix, mins = lu
                exp = np.array([(st[i][0] / st[i][1]) if i in st and st[i][1] > 0.5 else 10.0 for i in ix])
                share = exp / exp.sum() * 5.0
                po, pd_ = float((coef_o[ix] * share).sum()), float((coef_d[ix] * share).sum())
                u = usual.get(team)
                res[side] = (po, pd_, po - u[0] / u[2] if u else 0.0, pd_ - u[1] / u[2] if u else 0.0, team)
                todo.append((team, po, pd_, ix, mins))
            out[gm.game_id] = (res["h"][0], res["h"][1], res["a"][0], res["a"][1], res["h"][2], res["h"][3], res["a"][2], res["a"][3])
        for team, po, pd_, ix, mins in todo:                              # update only after the whole day is scored
            u = usual.get(team, (0.0, 0.0, 0.0))
            usual[team] = (u[0] * P.P_DECAY + po, u[1] * P.P_DECAY + pd_, u[2] * P.P_DECAY + 1.0)
            for i, mn in zip(ix, mins):
                s = st.get(i, (0.0, 0.0))
                st[i] = (s[0] * P.P_DECAY + mn, s[1] * P.P_DECAY + 1.0)
    f = pd.DataFrame.from_dict(out, orient="index", columns=NEWF + DEVF).rename_axis("game_id").reset_index()
    return d.merge(f, on="game_id", how="left").fillna({c: 0.0 for c in NEWF + DEVF})


def walk(d, seasons, cols_m, cols_t):
    out, first = [], d["season"].min()
    for s in seasons:
        tr, te = (d["season"] < s) & (d["season"] > first), d["season"] == s
        if tr.sum() < 1000 or te.sum() == 0:
            continue
        t = d[te][["game_id", "date", "home", "away", "margin", "total"]].copy()
        t["pm"] = P.Ridged().fit(d.loc[tr, cols_m], d.loc[tr, "margin"]).predict(d.loc[te, cols_m]).values
        keep = P.FEATS + [c for c in cols_t if c not in P.FEATS]
        t["pt"] = P.TotalRidge(cols_t).fit(d.loc[tr, keep], d.loc[tr, "total"]).predict(d.loc[te, keep]).values
        out.append(t)
    return pd.concat(out) if out else None


def miss(r):
    return float((r["pm"] - r["margin"]).abs().mean()), float((r["pt"] - r["total"]).abs().mean())


def main():
    gp, hp = f"{P.DATA}/nba_games.csv", f"{P.DATA}/nba_lines_hist.csv"
    players = P.load_players()
    if players is None or not (os.path.exists(gp) and os.path.exists(hp)):
        log("Need data/nba_games.csv, data/nba_lines_hist.csv and the player box scores (python nba_fetch_players.py).")
        return 1
    g = P.load_games(gp)
    d, _ = P.history_table(g)
    seasons = sorted(d["season"].unique())
    have = P.seasons_with_lines(d) or set()
    test = [s for s in seasons if s in have and (d["season"] == s).sum() >= 1100 and s >= seasons[0] + 2][-P.N_TEST:]
    val = [s for s in seasons if s not in have and s >= seasons[0] + 2 and (d["season"] == s).sum() >= 1100]
    log(f"{len(d)} games; choosing the rating strength on seasons without closing lines {[int(x) for x in val]}; grading on {[int(x) for x in test]}")
    if not val or not test:
        log("Not enough seasons for a clean split.")
        return 1
    t0 = time.time()
    pl, pids, lineup, X, y, dates = prep(g, d, players)
    log(f"{len(pids)} players, {X.shape[0]} team-games in the rating fit ({time.time() - t0:.0f}s)")

    best, best_alpha, scores, cache = None, None, {}, {}
    for a in ALPHAS:
        da = lineup_features(d, pl, pids, lineup, X, y, dates, a)
        cache[a] = da
        r = walk(da, val, P.FEATS + NEWF + DEVF, P.FEATS_T + NEWF + DEVF)
        mm, mt = miss(r)
        scores[a] = mm + mt
        log(f"  rating strength alpha={a:g}: validation miss margin {mm:.3f}, total {mt:.3f}   ({time.time() - t0:.0f}s)")
    best_alpha = min(scores, key=scores.get)
    d2 = cache[best_alpha]
    log(f"  -> using alpha={best_alpha:g}")

    # the crude first-test availability, for comparison
    d2 = P.add_availability(d2.copy(), players)
    h = pd.read_csv(hp)
    h["date"] = pd.to_datetime(h["date"])
    h = h[["date", "home", "away", "home_spread", "total"]].rename(columns={"total": "mkt_total"})
    variants = {
        "ratings only (tuned settings)": (P.FEATS, P.FEATS_T),
        "+ crude game-score availability": (P.FEATS + P.AVAIL, P.FEATS_T + P.AVAIL),
        "+ player-rating lineup, vs usual": (P.FEATS + DEVF, P.FEATS_T + DEVF),
        "+ player-rating lineup, level": (P.FEATS + NEWF, P.FEATS_T + NEWF),
        "+ player-rating lineup, both": (P.FEATS + NEWF + DEVF, P.FEATS_T + NEWF + DEVF)}
    rows, keep = [], {}
    for name, (cm, ct) in variants.items():
        r = walk(d2, test, cm, ct).merge(h, on=["date", "home", "away"], how="inner")
        keep[name] = r
        se, te = r["pm"] - r["home_spread"], r["pt"] - r["mkt_total"]
        rows.append({"model": name, "games": len(r), "margin_miss": float((r["pm"] - r["margin"]).abs().mean()),
                     "total_miss": float((r["pt"] - r["total"]).abs().mean()),
                     "corr_spread": float(np.corrcoef(se, r["margin"] - r["home_spread"])[0, 1]),
                     "corr_total": float(np.corrcoef(te, r["total"] - r["mkt_total"])[0, 1])})
    r0 = keep["ratings only (tuned settings)"]
    mk_m, mk_t = float((r0["home_spread"] - r0["margin"]).abs().mean()), float((r0["mkt_total"] - r0["total"]).abs().mean())
    res = pd.DataFrame(rows)
    res["margin_gap"], res["total_gap"] = res["margin_miss"] - mk_m, res["total_miss"] - mk_t
    log(f"\nGRADED against the closing lines (market miss: margin {mk_m:.3f}, total {mk_t:.3f}; gap = model minus market, smaller is better)")
    log(res.round(3).to_string(index=False))
    bn = res.loc[(res["margin_gap"] + res["total_gap"]).idxmin(), "model"]
    r = keep[bn]
    log(f"\nBest overall: {bn}")
    log("  spread win rate betting its side against the closing line (need > 52.4%):")
    log(P.win_record(r["pm"] - r["home_spread"], r["margin"] - r["home_spread"], P.SPREAD_EDGES).round(3).to_string(index=False))
    log("  total win rate:")
    log(P.win_record(r["pt"] - r["mkt_total"], r["total"] - r["mkt_total"], P.TOTAL_EDGES).round(3).to_string(index=False))

    # the best and worst rated players at the end, as a sanity check
    n = len(pids)
    mdl = Ridge(alpha=best_alpha, solver="sparse_cg", max_iter=300).fit(X, y)
    names = pl.drop_duplicates("pid").set_index("pid")["name"]
    inv = {i: p for p, i in pids.items()}
    c = pd.DataFrame({"name": [names.get(inv[i], inv[i]) for i in range(n)], "offense": mdl.coef_[:n], "defense": -mdl.coef_[n:2 * n]})
    c["net"] = c["offense"] + c["defense"]
    log("\nSANITY CHECK, highest net rating over all seasons (points per 100 possessions for a player on the floor all game; should be stars):")
    log(c.sort_values("net", ascending=False).head(12).round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
