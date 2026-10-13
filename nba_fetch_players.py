"""Download NBA player box scores (the free sportsdataverse files on GitHub) into data/nba_players.parquet.

One row per player per game: minutes, points, rebounds, assists and the pieces of Hollinger's game score. The model uses it to
estimate how strong each team's lineup was in each game (who actually played). The file is rebuilt each run and is not
committed to the repo (it is large); data/*.parquet is git-ignored.
ESPN names a season by the year it ends; the "season" column here is the year it starts.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nba_players.parquet")
FIRST_END_YEAR = int(os.environ.get("NBA_FROM_END_YEAR", "2018"))
URLS = ["https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/player_box/parquet/player_box_{y}.parquet",
        "https://github.com/sportsdataverse/hoopR-nba-data/raw/main/nba/player_box/parquet/player_box_{y}.parquet"]
ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BK": "BKN", "CHO": "CHA"}
NBA = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
       "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"}
# each field -> column names it might have in the file (first match wins)
WANT = {"gid": ["game_id"], "date": ["game_date_time", "game_date"], "team": ["team_abbreviation"], "pid": ["athlete_id", "player_id"],
        "name": ["athlete_display_name", "athlete_name", "player_name"], "min": ["minutes"], "pts": ["points"],
        "fgm": ["field_goals_made"], "fga": ["field_goals_attempted"], "ftm": ["free_throws_made"], "fta": ["free_throws_attempted"],
        "oreb": ["offensive_rebounds"], "dreb": ["defensive_rebounds"], "ast": ["assists"], "stl": ["steals"], "blk": ["blocks"],
        "tov": ["turnovers", "total_turnovers"], "pf": ["fouls", "personal_fouls"]}


def current_end_year(today=None):
    t = today or pd.Timestamp.now(tz="America/New_York")
    return t.year + 1 if t.month >= 9 else t.year


def load_year(y):
    last = None
    for u in URLS:
        try:
            return pd.read_parquet(u.format(y=y))
        except Exception as e:
            last = e
    raise RuntimeError(str(last)[:150])


def to_players(raw, end_year):
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"].astype(str).isin(["2", "2.0", "Regular Season"])]
    cols = {}
    for k, names in WANT.items():
        hit = next((n for n in names if n in raw.columns), None)
        if hit is None:
            raise KeyError(f"no column for '{k}' (tried {names}); the file has: {sorted(raw.columns)}")
        cols[k] = hit
    r = raw[list(cols.values())].copy()
    r.columns = list(cols)
    r["team"] = r["team"].replace(ABBR)
    r = r[r["team"].isin(NBA)].copy()
    num = [c for c in WANT if c not in ("gid", "date", "team", "pid", "name")]
    r[num] = r[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    d = pd.to_datetime(r["date"], utc=True)
    if (d.dt.hour + d.dt.minute).sum() > 0:                       # a UTC timestamp: shift so late games keep their US date
        d = d - pd.Timedelta(hours=6)
    r["date"] = d.dt.strftime("%Y-%m-%d")
    r["gid"], r["pid"] = r["gid"].astype(str), r["pid"].astype(str)
    r["season"] = end_year - 1
    r["reb"] = r["oreb"] + r["dreb"]
    # Hollinger game score: a one-number box-score summary on roughly the scale of points
    r["gmsc"] = (r["pts"] + 0.4 * r["fgm"] - 0.7 * r["fga"] - 0.4 * (r["fta"] - r["ftm"]) + 0.7 * r["oreb"] + 0.3 * r["dreb"]
                 + r["stl"] + 0.7 * r["ast"] + 0.7 * r["blk"] - 0.4 * r["pf"] - r["tov"])
    return r[["gid", "season", "date", "team", "pid", "name", "min", "pts", "reb", "ast", "gmsc"]]


def main():
    parts, failed = [], 0
    for y in range(FIRST_END_YEAR, current_end_year() + 1):
        try:
            p = to_players(load_year(y), y)
            print(f"  {y - 1}-{str(y)[2:]}: {len(p)} player-games, {p['gid'].nunique()} games", flush=True)
            parts.append(p)
        except KeyError as e:
            print(f"  {y - 1}-{str(y)[2:]}: {e}")
            return 1
        except Exception as e:
            print(f"  {y - 1}-{str(y)[2:]}: not downloaded ({e})")
            failed += 1
            if failed >= 3:
                print("  Three downloads failed, so stopping.")
                break
    if not parts:
        print("No player data downloaded.")
        return 1
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"Saved {len(out)} player-games ({out['season'].nunique()} seasons) to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
