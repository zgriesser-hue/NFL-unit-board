"""Save today's NBA player prop lines (points, rebounds, assists) from ParlayAPI, so we can later grade our minutes/stat projections
against real lines. Appends to boards/nba_props.csv. Costs 3 credits per page (one page is usually enough); capped at MAX_PAGES.
Needs the GitHub secret PARLAY_API_KEY (the same key the NFL player model uses).
"""
import os
import sys
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "boards", "nba_props.csv")
BASE = "https://parlay-api.com/v1/sports/basketball_nba/props"
MARKETS = "player_points,player_rebounds,player_assists"
BOOKS = os.environ.get("PROP_BOOKS", "draftkings,fanduel")
MAX_PAGES = int(os.environ.get("PROP_MAX_PAGES", "3"))
KEEP = ["snap_time", "book", "player", "market", "line", "over_price", "under_price", "home", "away", "event_id", "commence"]


def log(m):
    print(m, flush=True)


def fetch(key):
    rows, offset, credits = [], 0, 0
    for _ in range(MAX_PAGES):
        for attempt in range(3):
            r = requests.get(BASE, headers={"X-API-Key": key}, timeout=60,
                             params={"markets": MARKETS, "bookmakers": BOOKS, "limit": 10000, "offset": offset})
            if r.status_code == 503:           # board degraded: not charged, wait and retry the same page
                time.sleep(6)
                continue
            break
        if r.status_code != 200:
            log(f"ParlayAPI returned {r.status_code}: {r.text[:200]}")
            break
        credits += 3
        body = r.json()
        rows += body if isinstance(body, list) else body.get("data", [])
        if str(r.headers.get("x-result-truncated", "")).lower() == "true":
            log("note: the response was flagged truncated by ParlayAPI")
        if str(r.headers.get("x-result-has-more", "")).lower() != "true":
            break
        offset = int(r.headers.get("x-next-offset", offset + 10000))
    return rows, credits


def tidy(rows):
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%MZ")
    out = []
    for x in rows:
        if str(x.get("period", "FULL")).upper() not in ("FULL", "UNKNOWN", "NONE", ""):
            continue                           # full-game props only
        if x.get("line") is None:
            continue
        out.append((now, x.get("bookmaker"), x.get("player"), x.get("market_key"), x.get("line"), x.get("over_price"),
                    x.get("under_price"), x.get("home_team"), x.get("away_team"), x.get("canonical_event_id"), x.get("commence_time")))
    return pd.DataFrame(out, columns=KEEP)


def main():
    key = os.environ.get("PARLAY_API_KEY", "").strip()
    if not key:
        log("PARLAY_API_KEY is not set (add it under Settings > Secrets and variables > Actions).")
        return 1
    rows, credits = fetch(key)
    df = tidy(rows)
    log(f"{len(rows)} rows returned, {len(df)} full-game NBA props kept, about {credits} credits used")
    if df.empty:
        log("Nothing to save (normal before the season starts).")
        return 0
    if len(df):
        log(df.groupby(["book", "market"]).size().to_string())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    log(f"appended to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
