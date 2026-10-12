# Forward test for the totals board: grade every saved PREGAME totals board against the closing total and the final score.
# Boards are grouped by the weekday they were built (Mon / Wed / Fri ...), so the Monday board is judged on its own.
#   * win rate: betting the model's side at the number the board saw (pushes excluded; break-even at -110 is 52.4%)
#   * CLV: how many points the line moved toward the model's side by close (positive = you beat the closing number)
# Closing total: last capture in boards/closing_lines.csv before kickoff (closing_snapshot.py); falls back to the nflverse line.
# Run from the repo root (the GitHub workflow does this). Writes forward_totals_games.csv and forward_totals_summary.csv.

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("NFL_ROOT", os.getcwd())
THRESHOLDS = [0, 1, 2, 3]


def load_boards():
    files = sorted(glob.glob(f"{ROOT}/boards/totals_board_20*_wk*_*.csv"))
    if not files:
        return None
    b = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ["kickoff_utc", "built_at_utc"]:
        b[c] = pd.to_datetime(b[c], utc=True)
    b = b[b["built_at_utc"] < b["kickoff_utc"]].copy()                       # pregame boards only
    b["slot"] = b["built_at_utc"].dt.tz_convert("America/New_York").dt.day_name().str[:3]
    b = b.sort_values("built_at_utc").groupby(["game_id", "slot"]).tail(1)   # latest pregame board per game per weekday
    b["season"] = b["game_id"].str[:4].astype(int)
    return b


def add_results(g):
    import nflreadpy as nfl
    sched = nfl.load_schedules(seasons=sorted(g["season"].unique().tolist())).to_pandas()
    res = sched[["game_id", "home_score", "away_score", "total_line"]].rename(columns={"total_line": "nfl_close_total"})
    g = g.merge(res, on="game_id", how="left")
    g = g[g["home_score"].notna()].copy()
    g["actual_total"] = g["home_score"] + g["away_score"]
    return g


def add_closing(g):
    """Closing total: the last pre-kickoff capture of the consensus total, else nflverse's closing line."""
    g["close_total"], g["close_src"] = g["nfl_close_total"], "nflverse"
    cl_path = f"{ROOT}/boards/closing_lines.csv"
    if not os.path.exists(cl_path):
        return g
    cl = pd.read_csv(cl_path)
    if "total" not in cl or cl["total"].notna().sum() == 0:
        return g
    cl["captured_utc"] = pd.to_datetime(cl["captured_utc"], utc=True)
    cl["kickoff_utc"] = pd.to_datetime(cl["kickoff_utc"], utc=True)
    cl = cl[(cl["captured_utc"] < cl["kickoff_utc"]) & cl["total"].notna()].sort_values("captured_utc")
    cl = cl.groupby(["home_team", "away_team", "kickoff_utc"], as_index=False).tail(1)
    cl = cl.rename(columns={"kickoff_utc": "close_kick", "total": "cap_total"})
    g = g.sort_values("kickoff_utc")
    m = pd.merge_asof(g, cl.sort_values("close_kick")[["home_team", "away_team", "close_kick", "cap_total"]],
                      left_on="kickoff_utc", right_on="close_kick", by=["home_team", "away_team"],
                      direction="nearest", tolerance=pd.Timedelta(days=1))
    got = m["cap_total"].notna().values
    g.loc[got, "close_total"] = m.loc[got, "cap_total"].values
    g.loc[got, "close_src"] = "captured"
    return g


def grade(g, line_col):
    """Adds the bet result and CLV for betting the model's side at g[line_col]."""
    out = g.copy()
    out["line"] = out[line_col]
    out["edge"] = out["fair_total"] - out["line"]
    side = np.sign(out["edge"])
    out["result"] = np.sign(out["actual_total"] - out["line"]) * side            # +1 win, -1 loss, 0 push
    out["clv_pts"] = (out["close_total"] - out["line"]) * side                   # + = the line moved toward the model's side
    return out[out["line"].notna()]


def summarize(g, basis, subset):
    rows = []
    for slot, gs in g.groupby("slot"):
        for thr in THRESHOLDS:
            sel = gs[gs["edge"].abs() >= thr]
            dec = sel[sel["result"] != 0]
            n = len(dec)
            wr = float((dec["result"] > 0).mean()) if n else np.nan
            has = sel["clv_pts"].notna()
            rows.append({"slot": slot, "basis": basis, "subset": subset, "edge_at_least": thr, "games": len(sel), "bets": n,
                         "win_rate": wr, "win_se": float(np.sqrt(0.25 / n)) if n else np.nan,
                         "roi_at_110": wr * 100 / 110 - (1 - wr) if n else np.nan,
                         "avg_clv_pts": float(sel.loc[has, "clv_pts"].mean()) if has.any() else np.nan,
                         "moved_toward_pct": float((sel.loc[has & (sel["clv_pts"] != 0), "clv_pts"] > 0).mean())
                         if (has & (sel["clv_pts"] != 0)).any() else np.nan})
    return rows


def main():
    b = load_boards()
    if b is None or b.empty:
        print("No pregame totals boards yet.")
        return
    g = add_closing(add_results(b))
    if g.empty:
        print("No finished games with a saved pregame totals board yet.")
        return
    flag = lambda c: g[c].fillna("").astype(str).str.startswith(("CHECK", "override"))
    g["qb_flag"] = flag("home_QB_note") | flag("away_QB_note")
    out_games = []
    rows = []
    for basis, col in [("consensus", "market_total"), ("draftkings", "dk_total")]:
        if col not in g:
            continue
        gg = grade(g, col)
        if gg.empty:
            continue
        gg["basis"] = basis
        out_games.append(gg)
        rows += summarize(gg, basis, "all games")
        rows += summarize(gg[~gg["qb_flag"]], basis, "no QB CHECK/override")
    if not out_games:
        print("No graded games had a market total on the board.")
        return
    pd.concat(out_games).drop(columns=["line"]).to_csv(f"{ROOT}/forward_totals_games.csv", index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(f"{ROOT}/forward_totals_summary.csv", index=False)

    gm = out_games[0]
    print(f"Graded totals: {len(gm)} board-games from {gm['game_id'].nunique()} games | closing source: "
          f"{gm['close_src'].value_counts().to_dict()}")
    show = summ[(summ["subset"] == "all games") & (summ["basis"] == "consensus")]
    print("Betting the model's side at the number the board saw (break-even at -110 is 52.4%); CLV = points the line moved toward you:")
    for _, r in show.iterrows():
        if r["games"] == 0:
            continue
        print(f"  {r['slot']} board, |edge| >= {int(r['edge_at_least'])}: {int(r['bets'])} bets, win {r['win_rate']:.1%} "
              f"(+/- {r['win_se']:.1%}), ROI {r['roi_at_110']:+.1%}, CLV {r['avg_clv_pts']:+.2f} pts, "
              f"line moved toward model {r['moved_toward_pct']:.0%}")


if __name__ == "__main__":
    main()
