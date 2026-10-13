"""NBA line history: a few snapshots a day of every upcoming game's spread and total, appended to boards/nba_lines.csv.

The point is to build your own history (opening line -> closing line, by book) so an NBA model can be tested honestly later.
Free sources have almost no historical NBA lines, so saving them from day one is the whole job here.

Each run keeps games that start within LOOKAHEAD_HOURS and have not started, one row per game per run.
Columns: consensus (median across US books) spread and total, DraftKings' numbers, and the best price at any book.
home_spread: positive = home team favored (same convention as the NFL board).
Needs the ODDS_API_KEY environment variable. Costs 2 Odds API credits per run (spreads + totals, one region).
"""
import os
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "boards", "nba_lines.csv")
LOOKAHEAD_HOURS = float(os.environ.get("NBA_LOOKAHEAD_HOURS", "36"))
SPORT = os.environ.get("NBA_SPORT_KEY", "basketball_nba")
OK_PRICE = (-150, 130)          # ignore stale or odd prices when taking the consensus
COLS = ["captured_utc", "kickoff_utc", "home_team", "away_team",
        "home_spread", "spread_books", "dk_home_spread",
        "best_home_point", "best_home_price", "best_home_book", "best_away_point", "best_away_price", "best_away_book",
        "total", "total_books", "dk_total",
        "best_over_line", "best_over_price", "best_over_book", "best_under_line", "best_under_price", "best_under_book"]


def parse_event(ev, now):
    """One row of consensus lines and best prices for an Odds API event, or None."""
    home, away = ev["home_team"], ev["away_team"]
    hs, aw, ov, un, dk_sp, dk_tot = [], [], [], [], np.nan, np.nan   # hs/aw: (point, price, book) of each side's spread
    for bk in ev.get("bookmakers", []):
        name = bk.get("title") or bk.get("key")
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                pt, pr = oc.get("point"), oc.get("price")
                if pt is None or pr is None or not OK_PRICE[0] <= pr <= OK_PRICE[1]:
                    continue
                rec = (float(pt), int(pr), name)
                if mk["key"] == "spreads":
                    (hs if oc["name"] == home else aw).append(rec)
                    if bk.get("key") == "draftkings" and oc["name"] == home:
                        dk_sp = -float(pt)
                elif mk["key"] == "totals":
                    (ov if oc["name"] == "Over" else un).append(rec)
                    if bk.get("key") == "draftkings" and oc["name"] == "Over":
                        dk_tot = float(pt)
    if not (hs or ov):
        return None
    best = lambda rows, hi: (max(rows, key=lambda r: (r[0], r[1])) if hi else min(rows, key=lambda r: (r[0], -r[1]))) if rows else (np.nan, np.nan, "")
    bh, ba = best(hs, True), best(aw, True)         # bettor wants the highest number on either side of a spread
    bo, bu = best(ov, False), best(un, True)        # lowest number on the Over, highest on the Under
    return {"captured_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kickoff_utc": pd.Timestamp(ev["commence_time"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_team": home, "away_team": away,
            "home_spread": -float(np.median([r[0] for r in hs])) if hs else np.nan, "spread_books": len(hs), "dk_home_spread": dk_sp,
            "best_home_point": bh[0], "best_home_price": bh[1], "best_home_book": bh[2],
            "best_away_point": ba[0], "best_away_price": ba[1], "best_away_book": ba[2],
            "total": float(np.median([r[0] for r in ov])) if ov else np.nan, "total_books": len(ov), "dk_total": dk_tot,
            "best_over_line": bo[0], "best_over_price": bo[1], "best_over_book": bo[2],
            "best_under_line": bu[0], "best_under_price": bu[1], "best_under_book": bu[2]}


def main():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        print("ODDS_API_KEY is not set; nothing captured.")
        return 0
    r = requests.get(f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds",
                     params={"apiKey": key, "regions": "us", "markets": "spreads,totals", "oddsFormat": "american"}, timeout=60)
    r.raise_for_status()
    left = r.headers.get("x-requests-remaining")
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for ev in r.json():
        hrs = (pd.Timestamp(ev["commence_time"]) - now).total_seconds() / 3600
        if not 0 < hrs <= LOOKAHEAD_HOURS:
            continue
        row = parse_event(ev, now)
        if row:
            rows.append(row)
    if left is not None:
        print(f"Odds API credits remaining this period: {left}")
    if not rows:
        print(f"No NBA games start within the next {LOOKAHEAD_HOURS:g} hours (or no lines posted); nothing captured.")
        return 0
    new = pd.DataFrame(rows, columns=COLS)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    old = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    pd.concat([old, new], ignore_index=True)[COLS].to_csv(OUT, index=False)
    for x in rows:
        print(f"  {x['away_team']} @ {x['home_team']}: home spread {x['home_spread']:+.1f}, total {x['total']} "
              f"({x['spread_books']} books), tip {x['kickoff_utc']}")
    print(f"Captured {len(rows)} games.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
