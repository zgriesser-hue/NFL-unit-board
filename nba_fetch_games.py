"""Download NBA regular-season box scores (free sportsdataverse files on GitHub) into data/nba_games.csv.

stats.nba.com blocks cloud servers (GitHub Actions and Colab both time out), so this reads the same game box scores that the
sportsdataverse project publishes on GitHub. One row per game: date, teams, points, and the box-score pieces needed to
estimate possessions (field-goal attempts, free-throw attempts, offensive rebounds, turnovers). Past seasons are kept as saved;
only the current season (and any missing season) is downloaded, so a daily run is one or two small files.
ESPN names a season by the year it ends; the "season" column here is the year it starts (2025 = the 2025-26 season).
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nba_games.csv")
FIRST_END_YEAR = int(os.environ.get("NBA_FROM_END_YEAR", "2018"))      # 2018 = the 2017-18 season
ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS", "PHO": "PHX", "BK": "BKN", "CHO": "CHA"}
NBA = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
       "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"}     # drops All-Star and exhibition teams
COLS = ["game_id", "season", "date", "home", "away", "home_pts", "away_pts", "minutes",
        "home_fga", "home_fta", "home_oreb", "home_tov", "away_fga", "away_fta", "away_oreb", "away_tov"]
URLS = ["https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/team_box/parquet/team_box_{y}.parquet",
        "https://github.com/sportsdataverse/hoopR-nba-data/raw/main/nba/team_box/parquet/team_box_{y}.parquet"]


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


def to_games(raw, end_year):
    """ESPN team rows (one per team per game) -> one row per game."""
    if "season_type" in raw.columns:
        raw = raw[raw["season_type"].astype(str).isin(["2", "2.0", "Regular Season"])]
    c = dict(gid="game_id", date="game_date_time" if "game_date_time" in raw.columns else "game_date", abbr="team_abbreviation",
             ha="team_home_away", pts="team_score", fga="field_goals_attempted", fta="free_throws_attempted",
             oreb="offensive_rebounds", tov="total_turnovers" if "total_turnovers" in raw.columns else "turnovers")
    r = raw[list(c.values())].copy()
    r.columns = list(c)
    r["abbr"] = r["abbr"].replace(ABBR)
    h, a = r[r["ha"] == "home"].add_prefix("h_"), r[r["ha"] == "away"].add_prefix("a_")
    g = h.merge(a, left_on="h_gid", right_on="a_gid")
    g = g[g["h_abbr"].isin(NBA) & g["a_abbr"].isin(NBA)]
    d = pd.to_datetime(g["h_date"], utc=True)
    if (d.dt.hour + d.dt.minute).sum() > 0:              # a UTC timestamp: shift back so late games keep their US calendar date
        d = d - pd.Timedelta(hours=6)
    out = pd.DataFrame({"game_id": g["h_gid"].astype(str), "season": end_year - 1, "date": d.dt.strftime("%Y-%m-%d"),
                        "home": g["h_abbr"], "away": g["a_abbr"], "home_pts": g["h_pts"], "away_pts": g["a_pts"], "minutes": 240,
                        "home_fga": g["h_fga"], "home_fta": g["h_fta"], "home_oreb": g["h_oreb"], "home_tov": g["h_tov"],
                        "away_fga": g["a_fga"], "away_fta": g["a_fta"], "away_oreb": g["a_oreb"], "away_tov": g["a_tov"]})
    return out[COLS]


def main():
    old = pd.read_csv(OUT, dtype={"game_id": str}) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    have = set(old["season"].astype(int) + 1) if len(old) else set()
    now = current_end_year()
    todo = [y for y in range(FIRST_END_YEAR, now + 1) if y not in have or y == now]
    parts, failed = [old[~(old["season"].astype(int) + 1).isin(todo)]], 0
    for y in todo:
        try:
            g = to_games(load_year(y), y)
            print(f"  {y - 1}-{str(y)[2:]}: {len(g)} games")
            parts.append(g)
        except Exception as e:                           # not published yet (new season) or a network problem: keep what we had
            print(f"  {y - 1}-{str(y)[2:]}: not downloaded ({e})")
            parts.append(old[(old["season"].astype(int) + 1) == y])
            failed += 1
            if failed >= 3:
                print("  Three downloads failed, so stopping. Keeping the saved file.")
                break
    out = pd.concat(parts, ignore_index=True).drop_duplicates("game_id").sort_values(["date", "game_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[COLS].to_csv(OUT, index=False)
    print(f"Saved {len(out)} games ({out['season'].nunique()} seasons, {len(set(out['home']))} teams) to {OUT}")
    return 0 if len(out) else 1


if __name__ == "__main__":
    sys.exit(main())
