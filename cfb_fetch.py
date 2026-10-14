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
                     "home_conf": pick(x, "homeConference", "home_conference"), "away_conf": pick(x, "awayConference", "away_conference"),
                     "venue_id": pick(x, "venueId", "venue_id"),
                     **L.get(gid, {})})
    return pd.DataFrame(rows)


STATS_OUT = os.path.join(HERE, "data", "cfb_stats.csv")
VENUES_OUT = os.path.join(HERE, "data", "cfb_venues.csv")


def flat(x, *path):
    for k in path:
        x = x.get(k) if isinstance(x, dict) else None
        if x is None:
            return np.nan
    return x


def season_stats(y):
    """Game-level advanced team stats (plays, drives, expected points added per play, success rate) for one season."""
    rows = []
    for st in ("regular", "postseason"):
        try:
            data = get("/stats/game/advanced", year=y, seasonType=st)
        except SystemExit:
            data = []
        if not data and st == "regular":                          # some versions need a week
            for wk in range(1, 17):
                try:
                    data += get("/stats/game/advanced", year=y, week=wk, seasonType=st)
                except SystemExit:
                    break
        for x in data:
            o, d = x.get("offense") or {}, x.get("defense") or {}
            rows.append({"game_id": pick(x, "gameId", "game_id", "id"), "team": x.get("team"), "season": y,
                         "plays": o.get("plays"), "drives": o.get("drives"), "o_ppa": o.get("ppa"), "o_sr": o.get("successRate"),
                         "o_expl": o.get("explosiveness"), "d_ppa": d.get("ppa"), "d_sr": d.get("successRate"), "d_expl": d.get("explosiveness")})
    return pd.DataFrame(rows)


def venues():
    v = get("/venues")
    if v:
        log(f"  sample venue keys: {sorted(v[0].keys())}")
    rows = []
    for x in v:
        loc = x.get("location") or {}
        a, b = pick(x, "latitude", default=loc.get("y")), pick(x, "longitude", default=loc.get("x"))
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        lat, lon = (a, b) if abs(b) > abs(a) else (b, a)           # US latitudes are 24-50, longitudes 66-125 in size; sort out any mix-up
        rows.append({"venue_id": x.get("id"), "venue": x.get("name"), "lat": lat, "lon": lon, "dome": bool(pick(x, "dome", default=False)),
                     "elevation": pick(x, "elevation")})
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
    v = venues()
    v.to_csv(VENUES_OUT, index=False)
    log(f"saved {len(v)} venues with coordinates ({int(v['dome'].sum())} domes)")
    sp = []
    for y in range(FIRST, current_season() + 1):
        st = season_stats(y)
        log(f"  {y}: {len(st)} team-game stat rows ({st['plays'].notna().sum() if len(st) else 0} with plays)")
        sp.append(st)
    st = pd.concat(sp, ignore_index=True)
    st.to_csv(STATS_OUT, index=False)
    log(f"saved {len(st)} team-game stat rows to {STATS_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
