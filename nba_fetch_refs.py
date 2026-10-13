"""Download the three referees for each NBA game (ESPN's free game summary) into data/nba_refs.csv.

The file is a cache: games already in it are skipped, so each run picks up where the last one stopped. Newest seasons first.
Each run stops after MAX_MIN minutes (default 35) or after a run of failures, and saves as it goes. Run it again until it says "complete".
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES, OUT = os.path.join(HERE, "data", "nba_games.csv"), os.path.join(HERE, "data", "nba_refs.csv")
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
MAX_MIN = float(os.environ.get("REFS_MAX_MIN", "35"))
FROM_SEASON = int(os.environ.get("REFS_FROM_SEASON", "2018"))
WORKERS = int(os.environ.get("REFS_WORKERS", "10"))


def one(gid):
    for a in range(3):
        try:
            r = requests.get(URL.format(gid), timeout=30)
            if r.status_code == 200:
                off = (r.json().get("gameInfo") or {}).get("officials") or []
                names = [o.get("fullName") or o.get("displayName") for o in sorted(off, key=lambda o: o.get("order", 9))]
                return gid, (names + [None] * 3)[:3]
            if r.status_code in (404, 400):
                return gid, [None, None, None]                    # no summary for this game; remember so we do not ask again
        except Exception:
            pass
        time.sleep(1.5 * (a + 1))
    return gid, "fail"


def main():
    g = pd.read_csv(GAMES, dtype={"game_id": str})
    g = g[g["season"] >= FROM_SEASON].sort_values("date", ascending=False)
    have = pd.read_csv(OUT, dtype={"game_id": str}) if os.path.exists(OUT) else pd.DataFrame(columns=["game_id", "ref1", "ref2", "ref3"])
    todo = [x for x in g["game_id"] if x not in set(have["game_id"])]
    print(f"{len(have)} games already saved, {len(todo)} to fetch (seasons {FROM_SEASON}+)", flush=True)
    if not todo:
        print("complete")
        return 0
    t0, rows, fails, done = time.time(), [], 0, 0

    def save():
        pd.concat([have, pd.DataFrame(rows, columns=["game_id", "ref1", "ref2", "ref3"])]).drop_duplicates("game_id").to_csv(OUT, index=False)

    with ThreadPoolExecutor(WORKERS) as ex:
        for i in range(0, len(todo), 200):
            if time.time() - t0 > MAX_MIN * 60:
                print("time limit reached for this run; run it again to continue", flush=True)
                break
            for gid, res in ex.map(one, todo[i:i + 200]):
                if res == "fail":
                    fails += 1
                else:
                    rows.append((gid, *res))
                    fails = 0
                done += 1
            save()
            print(f"  {done}/{len(todo)} fetched ({time.time() - t0:.0f}s)", flush=True)
            if fails >= 40:
                print("many requests failing in a row (ESPN may be blocking this runner); stopping", flush=True)
                break
    save()
    left = len(todo) - len(rows)
    print("complete" if left <= 0 else f"{left} games still to fetch")
    ok = sum(1 for r in rows if r[1])
    print(f"{ok} of {len(rows)} fetched games had referees listed")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
