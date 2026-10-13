"""Download NBA regular-season team game logs (free, via the nba_api package) into data/nba_games.csv.

One row per game: date, teams, points, and the box-score pieces needed to estimate possessions
(field-goal attempts, free-throw attempts, offensive rebounds, turnovers). Past seasons are kept as saved
and only the current season (plus any missing season) is downloaded, so daily runs are one or two requests.

stats.nba.com sometimes refuses or times out on cloud servers. If that happens the old file is kept and the
run says so; you can then run this script on your own computer (pip install nba_api pandas) and upload data/nba_games.csv.
"""
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "nba_games.csv")
FIRST_SEASON = int(os.environ.get("NBA_FROM", "2017"))       # 2017 = the 2017-18 season
COLS = ["game_id", "season", "date", "home", "away", "home_pts", "away_pts", "minutes",
        "home_fga", "home_fta", "home_oreb", "home_tov", "away_fga", "away_fta", "away_oreb", "away_tov"]


def season_label(y):
    return f"{y}-{str(y + 1)[2:]}"


def current_season(today=None):
    t = today or pd.Timestamp.now(tz="America/New_York")
    return t.year if t.month >= 9 else t.year - 1


def fetch_season(y, tries=2):
    from nba_api.stats.endpoints import leaguegamelog
    last = None
    for i in range(tries):
        try:
            raw = leaguegamelog.LeagueGameLog(season=season_label(y), season_type_all_star="Regular Season",
                                              player_or_team_abbreviation="T", timeout=45).get_data_frames()[0]
            return to_games(raw, y)
        except Exception as e:                         # blocked, timed out, or not published yet
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(str(last)[:150])


def to_games(raw, y):
    """Team-game rows -> one row per game. Home rows say 'AAA vs. BBB', away rows 'AAA @ BBB'."""
    raw = raw.copy()
    raw["is_home"] = raw["MATCHUP"].str.contains("vs.", regex=False)
    keep = ["GAME_ID", "TEAM_ABBREVIATION", "PTS", "FGA", "FTA", "OREB", "TOV", "MIN"]
    h = raw[raw["is_home"]][keep + ["GAME_DATE"]].add_prefix("h_")
    a = raw[~raw["is_home"]][keep].add_prefix("a_")
    g = h.merge(a, left_on="h_GAME_ID", right_on="a_GAME_ID")
    return pd.DataFrame({"game_id": g["h_GAME_ID"], "season": y, "date": pd.to_datetime(g["h_GAME_DATE"]).dt.strftime("%Y-%m-%d"),
                         "home": g["h_TEAM_ABBREVIATION"], "away": g["a_TEAM_ABBREVIATION"],
                         "home_pts": g["h_PTS"], "away_pts": g["a_PTS"], "minutes": g["h_MIN"],
                         "home_fga": g["h_FGA"], "home_fta": g["h_FTA"], "home_oreb": g["h_OREB"], "home_tov": g["h_TOV"],
                         "away_fga": g["a_FGA"], "away_fta": g["a_FTA"], "away_oreb": g["a_OREB"], "away_tov": g["a_TOV"]})[COLS]


def main():
    old = pd.read_csv(OUT, dtype={"game_id": str}) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    have = set(old["season"].astype(int)) if len(old) else set()
    now = current_season()
    todo = [y for y in range(FIRST_SEASON, now + 1) if y not in have or y == now]
    parts, failed = [old[~old["season"].isin(todo)]], []
    for y in todo:
        try:
            df = fetch_season(y)
            print(f"  {season_label(y)}: {len(df)} games")
            parts.append(df)
        except Exception as e:
            print(f"  {season_label(y)}: could not download ({e})")
            failed.append(y)
            parts.append(old[old["season"] == y])      # keep what we had
            if len(failed) >= 2:                       # blocked or down: stop instead of waiting through every season
                print("  Two downloads failed in a row, so stopping. Keeping the saved file.")
                parts.extend(old[old["season"] == z] for z in todo if z > y)
                break
        time.sleep(1.5)
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates("game_id").sort_values(["date", "game_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[COLS].to_csv(OUT, index=False)
    print(f"Saved {len(out)} games ({out['season'].nunique()} seasons) to {OUT}")
    if len(out) == 0:
        print("No data. Run this script on your own computer and upload data/nba_games.csv.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
