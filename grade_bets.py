"""Grades the bets in bets.csv against final scores (boards/results_latest.csv) and writes boards/bets_graded_latest.csv.

bets.csv is the permanent record of your bets (the Bets tab loads it automatically). Columns match the app: placed, game_id, season,
week, away_team, home_team, pick, line, odds, stake, book, note, model_home, market_home, type (Spread / Moneyline / Total / Prop),
desc, manual_result. Props are not graded automatically: put Win, Loss or Push in manual_result.
Spreads: line is for your team (-2.5 = laying 2.5). Totals: pick is Over/Under and line is the total. Moneyline: odds only.
Profit is in the same units as stake. CLV (spread bets only) uses the last line captured before kickoff in boards/closing_lines.csv, if present.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NFL_ROOT", HERE)


def profit(stake, odds, outcome):
    if outcome == "Win":
        return stake * (100.0 / abs(odds) if odds < 0 else odds / 100.0)
    return -stake if outcome == "Loss" else 0.0


def main():
    p = f"{ROOT}/bets.csv"
    if not os.path.exists(p):
        print("No bets.csv; nothing to grade.")
        return 0
    b = pd.read_csv(p)
    if b.empty:
        print("bets.csv is empty.")
        return 0
    res = pd.read_csv(f"{ROOT}/boards/results_latest.csv")[["game_id", "home_score", "away_score"]] if os.path.exists(f"{ROOT}/boards/results_latest.csv") else pd.DataFrame(columns=["game_id", "home_score", "away_score"])
    b = b.merge(res, on="game_id", how="left")
    for c in ("line", "odds", "stake"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    typ = b["type"].fillna("Spread")
    is_home = b["pick"] == b["home_team"]
    pm = np.where(is_home, b["home_score"] - b["away_score"], b["away_score"] - b["home_score"])
    pend = b["home_score"].isna()
    spread = pm + b["line"]
    tot = b["home_score"] + b["away_score"]
    ou = np.where(b["pick"].eq("Over"), tot - b["line"], b["line"] - tot)
    wl = lambda x: np.where(x > 0, "Win", np.where(x < 0, "Loss", "Push"))
    r = np.where(pend, "Pending", wl(spread))
    r = np.where(typ.eq("Moneyline"), np.where(pend, "Pending", wl(pm)), r)
    r = np.where(typ.eq("Total"), np.where(pend, "Pending", wl(ou)), r)
    man = b["manual_result"].fillna("").astype(str).str.strip().str.title()
    r = np.where(typ.eq("Prop"), np.where(man.isin(["Win", "Loss", "Push"]), man, "Pending"), r)
    b["result"] = r
    b["profit"] = [profit(s, o, x) if x != "Pending" else np.nan for s, o, x in zip(b["stake"], b["odds"], b["result"])]
    b["clv"] = np.nan
    cp = f"{ROOT}/boards/closing_lines.csv"
    if os.path.exists(cp):
        c = pd.read_csv(cp)
        c["captured_utc"] = pd.to_datetime(c["captured_utc"], utc=True)
        c["kickoff_utc"] = pd.to_datetime(c["kickoff_utc"], utc=True)
        c = c[c["captured_utc"] < c["kickoff_utc"]].sort_values("captured_utc")
        c["season"] = np.where(c["kickoff_utc"].dt.month >= 8, c["kickoff_utc"].dt.year, c["kickoff_utc"].dt.year - 1)
        c = c.groupby(["season", "home_team", "away_team"], as_index=False).tail(1)[["season", "home_team", "away_team", "home_spread"]]
        b["season"] = pd.to_numeric(b["season"], errors="coerce")
        b = b.merge(c.rename(columns={"home_spread": "close_home"}), on=["season", "home_team", "away_team"], how="left")
        close_pick = np.where(b["pick"] == b["home_team"], -b["close_home"], b["close_home"])
        b["clv"] = np.where(b["type"].fillna("Spread").eq("Spread"), b["line"] - close_pick, np.nan)
        b = b.drop(columns=["close_home"])
    os.makedirs(f"{ROOT}/boards", exist_ok=True)
    b.to_csv(f"{ROOT}/boards/bets_graded_latest.csv", index=False)
    done = b[b["result"] != "Pending"]
    print(f"{len(b)} bets: {int((done['result'] == 'Win').sum())}-{int((done['result'] == 'Loss').sum())}-{int((done['result'] == 'Push').sum())}, "
          f"{int((b['result'] == 'Pending').sum())} pending, profit {done['profit'].sum():+.2f} on {done['stake'].sum():g} risked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
