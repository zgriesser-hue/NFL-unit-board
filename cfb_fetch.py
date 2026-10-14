"""Download college football games and betting lines from CollegeFootballData.com into data/cfb_games.csv.

One row per game: teams, score, date, neutral site, and the market lines (median across the books the site lists: closing and opening
spread and total). Needs the free API key in the environment variable CFBD_API_KEY (a GitHub secret in the workflow).
About 2 calls per season, so a full download is roughly 25 calls.
Spread sign: the site gives the HOME team's spread (negative = home favored). cfb_model_pipeline.py double-checks the sign from the data.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "cfb_games.csv")
BASES = [b for b in (os.environ.get("CFBD_BASE"), "https://api.collegefootballdata.com", "https://apinext.collegefootballdata.com") if b]
FIRST = int(os.environ.get("CFB_FROM", "2015"))
KEY = os.environ.get("CFBD_API_KEY", "").strip()
_base = {}


def log(m):
    print(m, flush=True)


def get(path, **params):
    last = None
    for base in ([_base["b"]] if "b" in _base else BASES):
        for attempt in range(3):
            try:
                r = requests.get(base + path, params=params, headers={"Authorization": f"Bearer {KEY}", "Accept": "application/json"}, timeout=90)
            except Exception as e:
                last = f"{base}: {str(e)[:100]}"
                time.sleep(2)
                continue
            if r.status_code == 200:
                _base["b"] = base
                return r.json()
            if r.status_code in (401, 403):
                raise SystemExit(f"{r.status_code} from {base}{path}: the key was refused. Check the CFBD_API_KEY secret (no spaces or quotes). {r.text[:150]}")
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            last = f"{base}{path} -> {r.status_code} {r.text[:150]}"
            break
    raise SystemExit(f"could not fetch {path} {params}: {last}")


def pick(d, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default


def current_season(today=None):
    t = today or pd.Timestamp.now(tz="America/New_York")
    return t.year if t.month >= 8 else t.year - 1


def one_season(y):
    games = get("/games", year=y, seasonType="both")
    lines = get("/lines", year=y, seasonType="both")
    if y == FIRST and games:
        log(f"  sample game keys: {sorted(games[0].keys())}")
        if lines:
            log(f"  sample lines keys: {sorted(lines[0].keys())}; book keys: {sorted((lines[0].get('lines') or [{}])[0].keys())}")
    L = {}
    for x in lines:
        gid = pick(x, "id", "gameId")
        books = x.get("lines") or []
        sp = [pick(b, "spread") for b in books]
        so = [pick(b, "spreadOpen") for b in books]
        ou = [pick(b, "overUnder") for b in books]
        oo = [pick(b, "overUnderOpen") for b in books]
        med = lambda v: float(np.median([float(a) for a in v if a is not None])) if any(a is not None for a in v) else np.nan
        L[gid] = {"spread_close": med(sp), "spread_open": med(so), "total_close": med(ou), "total_open": med(oo), "n_books": len(books)}
    rows = []
    for x in games:
        gid = pick(x, "id")
        rows.append({"game_id": gid, "season": pick(x, "season", default=y), "week": pick(x, "week"), "season_type": pick(x, "seasonType", "season_type"),
                     "date": pick(x, "startDate", "start_date"), "neutral": bool(pick(x, "neutralSite", "neutral_site", default=False)),
                     "home": pick(x, "homeTeam", "home_team"), "away": pick(x, "awayTeam", "away_team"),
                     "home_pts": pick(x, "homePoints", "home_points"), "away_pts": pick(x, "awayPoints", "away_points"),
                     "home_class": pick(x, "homeClassification", "home_division"), "away_class": pick(x, "awayClassification", "away_division"),
                     **L.get(gid, {})})
    return pd.DataFrame(rows)


def main():
    if not KEY:
        log("CFBD_API_KEY is not set (add it under Settings > Secrets and variables > Actions).")
        return 1
    parts = []
    for y in range(FIRST, current_season() + 1):
        df = one_season(y)
        done = df["home_pts"].notna().sum()
        log(f"  {y}: {len(df)} games ({done} played), {df['spread_close'].notna().sum()} with a closing spread, {df['total_close'].notna().sum()} with a closing total  [{_base.get('b')}]")
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    log(f"saved {len(out)} games to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
