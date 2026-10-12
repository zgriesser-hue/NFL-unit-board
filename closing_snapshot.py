"""Closing-line snapshot for the unit board: the market spread shortly before kickoff.

Run on a schedule a couple of hours before each kickoff window (see .github/workflows/closing.yml). One Odds API
credit per run. Keeps only games that start within CLOSE_WINDOW_MIN minutes and have not started, and appends
one row per game per run to boards/closing_lines.csv. The closing line for a game is its LAST row before kickoff.

home_spread: median spread across US books, positive = home team favored (same convention as the board).
Needs the ODDS_API_KEY environment variable (the same key the weekly board uses).
"""
import os
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "boards", "closing_lines.csv")
WINDOW_MIN = int(os.environ.get("CLOSE_WINDOW_MIN", "180"))
COLS = ["captured_utc", "kickoff_utc", "home_team", "away_team", "home_spread", "books", "total", "total_books", "dk_total"]

TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS", "Washington Football Team": "WAS"}


def main():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY is not set; nothing captured.")
        return 0
    r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
                     params={"apiKey": key, "regions": "us", "markets": "spreads,totals", "oddsFormat": "american"}, timeout=60)
    r.raise_for_status()
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for ev in r.json():
        kick = pd.Timestamp(ev["commence_time"])
        mins = (kick - now).total_seconds() / 60
        if not (0 < mins <= WINDOW_MIN):
            continue
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        pts = [oc["point"] for bk in ev.get("bookmakers", []) for mk in bk.get("markets", []) if mk["key"] == "spreads"
               for oc in mk["outcomes"] if oc["name"] == ev["home_team"] and -150 <= oc.get("price", -110) <= 130]
        tpts = [oc["point"] for bk in ev.get("bookmakers", []) for mk in bk.get("markets", []) if mk["key"] == "totals"
                for oc in mk["outcomes"] if oc["name"] == "Over" and -150 <= oc.get("price", -110) <= 130]
        dk = [oc["point"] for bk in ev.get("bookmakers", []) if bk.get("key") == "draftkings" for mk in bk.get("markets", [])
              if mk["key"] == "totals" for oc in mk["outcomes"] if oc["name"] == "Over"]
        if home and away and (pts or tpts):
            rows.append({"captured_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "kickoff_utc": kick.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "home_team": home, "away_team": away,
                         "home_spread": -float(np.median(pts)) if pts else np.nan, "books": len(pts),
                         "total": float(np.median(tpts)) if tpts else np.nan, "total_books": len(tpts),
                         "dk_total": float(dk[0]) if dk else np.nan})
    if not rows:
        print(f"No games start within the next {WINDOW_MIN} minutes (or no spreads posted); nothing captured.")
        return 0
    new = pd.DataFrame(rows, columns=COLS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    pd.concat([old, new], ignore_index=True)[COLS].to_csv(OUT, index=False)
    for r_ in rows:
        print(f"  {r_['away_team']} @ {r_['home_team']}: home spread {r_['home_spread']:+.1f} ({r_['books']} books), "
              f"total {r_['total']} ({r_['total_books']} books), kickoff {r_['kickoff_utc']}")
    print(f"Captured {len(rows)} games.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
