"""Convert the Kaggle 'NBA Odds Data' file (oddsData.csv, scraped from sportsbookreviewsonline) to data/nba_lines_hist.csv.

Usage: python nba_convert_lines.py path/to/oddsData.csv
The source has one row per team per game: 'spread' is that team's handicap (negative = favorite) and 'total' is the game total.
Output: date, home, away, home_spread (positive = home team favored, our convention), total, with standard team abbreviations.
Only the home team's row is used, so each game appears once. Seasons before 2017-18 are left out (the game data starts there).
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nba_lines_hist.csv")
ABBR = {"Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "New Jersey": "BKN", "Charlotte": "CHA", "Chicago": "CHI", "Cleveland": "CLE",
        "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND", "LA Clippers": "LAC",
        "LA Lakers": "LAL", "Memphis": "MEM", "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN", "New Orleans": "NOP", "New York": "NYK",
        "Oklahoma City": "OKC", "Seattle": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "Phoenix": "PHX", "Portland": "POR",
        "Sacramento": "SAC", "San Antonio": "SAS", "Toronto": "TOR", "Utah": "UTA", "Washington": "WAS"}


def convert(src, first_date="2017-09-01"):
    d = pd.read_csv(src)
    h = d[d["home/visitor"] == "vs"].copy()
    h["home"], h["away"] = h["team"].map(ABBR), h["opponent"].map(ABBR)
    unknown = set(h.loc[h["home"].isna(), "team"]) | set(h.loc[h["away"].isna(), "opponent"])
    if unknown:
        raise SystemExit(f"Unknown team names: {sorted(unknown)}")
    h["home_spread"] = -h["spread"]                         # source: negative = this (home) team is favored; ours: positive = home favored
    out = h.loc[h["date"] >= first_date, ["date", "home", "away", "home_spread", "total"]]
    return out.drop_duplicates(["date", "home", "away"]).sort_values(["date", "home"]).reset_index(drop=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out = convert(sys.argv[1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} games ({out['date'].min()} to {out['date'].max()}) to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
