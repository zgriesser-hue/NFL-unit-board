"""Can we project NBA player minutes better when we know which teammates are out?  (the first building block for player props)

Uses data/nba_players.parquet (made by nba_fetch_players.py). For every player-game where the player played, it predicts his minutes
from information available before tip-off, in two ways:
  1. BASELINE: his own recent minutes only (recency-weighted average, last game, season average, rest).
  2. WITH TEAMMATES OUT: the baseline plus how many minutes of regular rotation players are missing from that game.
Walk-forward: each season is predicted by a model trained only on earlier seasons. Smaller error is better.
Caveat: "who is out" is taken from who did not play, which is what an injury report would tell us on game day (minus late changes and
the game-time decisions). It also only scores players who actually played; the chance of a player sitting out is a separate piece.
"""
import os
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "boards", "nba_minutes_latest.csv")
WIN = 10          # a "rotation player" is judged on the last 10 games the team played
HALF = 5.0        # half-life (in games played) of the recency-weighted minutes average


def log(m):
    print(m, flush=True)


def load():
    for name, rd in (("nba_players.parquet", pd.read_parquet), ("nba_players.csv.gz", pd.read_csv)):
        p = os.path.join(DATA, name)
        if os.path.exists(p):
            d = rd(p, dtype={"gid": str, "pid": str}) if name.endswith("gz") else rd(p)
            d["pid"], d["gid"] = d["pid"].astype(str), d["gid"].astype(str)
            return d
    return None


def build(pl):
    """One row per player-game (played), features computed from games strictly before it."""
    pl = pl[pl["min"] > 0].copy()
    pl["date"] = pd.to_datetime(pl["date"])
    rows = []
    for team, t in pl.groupby("team"):
        games = t.groupby("gid")["date"].first().sort_values(kind="stable")
        order = list(games.index)
        dates = games.values
        by_game = {g: grp for g, grp in t.groupby("gid")}
        hist = defaultdict(list)            # pid -> minutes in games he played, in order
        seen_in = defaultdict(deque)        # pid -> deque of (game index) played within the window (by team-game index)
        season_of = {}
        last_season = None
        for i, g in enumerate(order):
            grp = by_game[g]
            season = int(grp["season"].iloc[0])
            if season != last_season:
                last_season = season
                szn_min = defaultdict(list)
            lo = i - WIN
            # regular rotation going into this game: played in >= 5 of the last 10 team games, averaging >= 14 minutes then
            rot = {}
            for pid, dq in seen_in.items():
                while dq and dq[0][0] < lo:
                    dq.popleft()
                if len(dq) >= 5:
                    m = np.mean([x[1] for x in dq])
                    if m >= 14:
                        rot[pid] = m
            here = set(grp["pid"])
            out_pids = [p for p in rot if p not in here]
            out_min = float(sum(rot[p] for p in out_pids))
            top_out = float(max([rot[p] for p in out_pids], default=0.0))
            rest = (dates[i] - dates[i - 1]) / np.timedelta64(1, "D") if i > 0 else 3.0
            for r in grp.itertuples():
                h = hist[r.pid]
                if len(h) >= 5:
                    w = 0.5 ** (np.arange(len(h))[::-1] / HALF)
                    ew = float(np.dot(w, h) / w.sum())
                    s = szn_min[r.pid]
                    rows.append((g, season, dates[i], team, r.pid, r.min, ew, h[-1], float(np.mean(h[-10:])),
                                 float(np.mean(s)) if s else ew, len(h), min(rest, 7.0), float(rest <= 1.0),
                                 out_min, len(out_pids), top_out, out_min * ew / 30.0))
            for r in grp.itertuples():
                hist[r.pid].append(r.min)
                szn_min[r.pid].append(r.min)
                seen_in[r.pid].append((i, r.min))
    cols = ["gid", "season", "date", "team", "pid", "min", "ew", "last", "avg10", "szn", "n_hist", "rest", "b2b",
            "out_min", "n_out", "top_out", "out_x_ew"]
    return pd.DataFrame(rows, columns=cols)


BASE = ["ew", "last", "avg10", "szn", "n_hist", "rest", "b2b"]
WITH = BASE + ["out_min", "n_out", "top_out", "out_x_ew"]


def fit(X, y):
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, max_depth=4, min_samples_leaf=60, random_state=0)
    return m.fit(X, y)


def main():
    pl = load()
    if pl is None:
        log("No player data found (data/nba_players.parquet). Run nba_fetch_players.py first.")
        return 1
    log(f"{len(pl)} player-games loaded; building pregame features ...")
    d = build(pl)
    log(f"{len(d)} player-games with 5+ games of history, {d['season'].nunique()} seasons")
    seasons = sorted(d["season"].unique())
    parts = []
    for s in seasons[2:]:
        tr, te = d[d["season"] < s], d[d["season"] == s].copy()
        if len(tr) < 5000:
            continue
        te["p_base"] = fit(tr[BASE], tr["min"]).predict(te[BASE])
        te["p_with"] = fit(tr[WITH], tr["min"]).predict(te[WITH])
        parts.append(te)
    if not parts:
        log("Not enough seasons to test.")
        return 1
    r = pd.concat(parts)
    r["e_naive"] = (r["min"] - r["ew"]).abs()
    r["e_base"] = (r["min"] - r["p_base"]).abs()
    r["e_with"] = (r["min"] - r["p_with"]).abs()
    log("")
    log(f"MINUTES PROJECTION TEST, {len(r)} player-games in seasons {[int(x) for x in sorted(r['season'].unique())]} (mean absolute error in minutes, smaller is better)")
    log(f"  recent average only:        {r['e_naive'].mean():.3f}")
    log(f"  baseline model:             {r['e_base'].mean():.3f}")
    log(f"  + teammates out:            {r['e_with'].mean():.3f}   ({(1 - r['e_with'].mean() / r['e_base'].mean()) * 100:+.1f}% vs baseline)")
    rot = r[r["ew"] >= 14]
    log("")
    log(f"  rotation players only ({len(rot)}): baseline {rot['e_base'].mean():.3f}, with teammates out {rot['e_with'].mean():.3f}")
    rows = []
    for name, m in (("no rotation player out", rot["out_min"] == 0), ("10-25 rotation minutes out", (rot["out_min"] > 0) & (rot["out_min"] <= 25)),
                    ("25+ rotation minutes out", rot["out_min"] > 25), ("a 28+ minute player out", rot["top_out"] >= 28)):
        x = rot[m]
        if len(x) >= 50:
            rows.append((name, len(x), x["e_base"].mean(), x["e_with"].mean(), (x["p_with"] - x["p_base"]).mean(), (x["min"] - x["ew"]).mean()))
    log("  by situation:      games  base_err  with_err  model_shift  actual_shift_vs_recent_avg")
    for n, c, b, w, sh, ac in rows:
        log(f"    {n:28s} {c:5d}   {b:6.3f}    {w:6.3f}     {sh:+6.2f}      {ac:+6.2f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r.drop(columns=["date"]).to_csv(OUT, index=False)
    log(f"\nsaved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
