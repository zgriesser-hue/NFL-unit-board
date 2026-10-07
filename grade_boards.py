# Grade the saved weekly boards against real results: the forward test.
# Only boards built BEFORE a game's kickoff count, and the latest such board is used for each game.
# Run from the repo root (the GitHub workflow does this).

import glob
import os
import numpy as np
import pandas as pd
import nflreadpy as nfl

ROOT = os.environ.get("NFL_ROOT", os.getcwd())
files = sorted(glob.glob(f"{ROOT}/boards/unit_board_20*_wk*_*.csv"))
if not files:
    raise SystemExit("No dated boards yet.")

b = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
for c in ["kickoff_utc", "built_at_utc"]:
    b[c] = pd.to_datetime(b[c], utc=True)
b = b[b["built_at_utc"] < b["kickoff_utc"]]                       # pregame boards only
b = b.sort_values("built_at_utc").groupby("game_id").tail(1)      # latest pregame version per game
b["season"] = b["game_id"].str[:4].astype(int)
b["week"] = b["game_id"].str[5:7].astype(int)

sched = nfl.load_schedules(seasons=sorted(b["season"].unique().tolist())).to_pandas()
g = b.merge(sched[["game_id", "home_score", "away_score"]], on="game_id", how="left")
g = g[g["home_score"].notna()].copy()
if g.empty:
    raise SystemExit("No finished games with a saved pregame board yet.")
g["margin"] = g["home_score"] - g["away_score"]
g.to_csv(f"{ROOT}/forward_test_games.csv", index=False)

mae = lambda p, y: float(np.abs(p - y).mean())
print(f"Graded games: {len(g)} | weeks: {sorted(g['week'].unique().tolist())}")
print(f"MAE vs actual margin: unit model {mae(g['fair_home_spread'], g['margin']):.2f}")

rows = []
if "market_home_spread" in g:
    h = g.dropna(subset=["market_home_spread"]).copy()
    if len(h):
        print(f"On the {len(h)} games with a market line: model {mae(h['fair_home_spread'], h['margin']):.2f} "
              f"| market {mae(h['market_home_spread'], h['margin']):.2f}")
        edge = h["fair_home_spread"] - h["market_home_spread"]
        cover = h["margin"] - h["market_home_spread"]
        print("Betting the model's side vs the market (break-even at -110 is 52.4%):")
        for thr in [0, 1, 2, 3]:
            sel = h[(edge.abs() >= thr) & (cover != 0)]
            if len(sel) == 0:
                continue
            win = np.sign(edge[sel.index]) == np.sign(cover[sel.index])
            n, wr = len(sel), float(win.mean())
            roi = wr * 100 / 110 - (1 - wr)
            print(f"  |edge| >= {thr}: {n} bets, win {wr:.1%} (+/- {np.sqrt(0.25 / n):.1%}), ROI {roi:+.1%}")
            rows.append({"edge_threshold": thr, "bets": n, "win_rate": wr, "roi_at_110": roi})
pd.DataFrame(rows).to_csv(f"{ROOT}/forward_test_summary.csv", index=False)
