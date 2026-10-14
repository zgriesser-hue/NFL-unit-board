"""Game-time weather for every college football game (free Open-Meteo archive, no key) into data/cfb_weather.csv.

Needs data/cfb_games.csv (with venue_id) and data/cfb_venues.csv from cfb_fetch.py. Indoor stadiums are marked and get no weather.
It is a cache: games already in the file are skipped, so reruns only fetch new games. One request per stadium per season.
Columns: game_id, indoor, temp_f, wind_mph, precip_in (average over the three hours after kickoff).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
GAMES, VENUES, OUT = (os.path.join(HERE, "data", f) for f in ("cfb_games.csv", "cfb_venues.csv", "cfb_weather.csv"))
MAX_MIN = float(os.environ.get("WX_MAX_MIN", "40"))


def log(m):
    print(m, flush=True)


def fetch(lat, lon, d0, d1):
    params = {"latitude": lat, "longitude": lon, "start_date": d0, "end_date": d1, "timezone": "UTC",
              "hourly": "temperature_2m,wind_speed_10m,precipitation", "temperature_unit": "fahrenheit",
              "wind_speed_unit": "mph", "precipitation_unit": "inch"}
    for a in range(4):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
            if r.status_code == 200:
                return r.json().get("hourly")
            if r.status_code == 429:
                time.sleep(20 * (a + 1))
                continue
            return None
        except Exception:
            time.sleep(3)
    return None


def main():
    g = pd.read_csv(GAMES)
    v = pd.read_csv(VENUES)
    g["kick"] = pd.to_datetime(g["date"], utc=True, errors="coerce")
    g = g[g["kick"].notna() & g["home_pts"].notna() & (g["spread_close"].notna() | g["total_close"].notna())].merge(v[["venue_id", "lat", "lon", "dome"]], on="venue_id", how="left")
    have = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["game_id", "indoor", "temp_f", "wind_mph", "precip_in"])
    todo = g[~g["game_id"].isin(set(have["game_id"]))]
    log(f"{len(have)} games already saved, {len(todo)} to fetch; {int(todo['lat'].isna().sum())} have no venue coordinates")
    rows, t0 = [], time.time()
    jobs = []
    for (vid, season), grp in todo.groupby(["venue_id", "season"], dropna=True):
        r0 = grp.iloc[0]
        if pd.isna(r0["lat"]):
            continue
        if bool(r0["dome"]):
            rows += [(x, 1, np.nan, np.nan, np.nan) for x in grp["game_id"]]
        else:
            jobs.append(grp)
    jobs.sort(key=lambda j: -int(j.iloc[0]["season"]))  # newest seasons first, so a partial run covers whole recent seasons
    log(f"{len(jobs)} weather requests to make ({len(rows)} indoor games need none)")

    def work(grp):
        r0 = grp.iloc[0]
        h = fetch(r0["lat"], r0["lon"], grp["kick"].min().strftime("%Y-%m-%d"), (grp["kick"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        out = []
        if h:
            t = pd.to_datetime(pd.Series(h["time"]), utc=True)
            df = pd.DataFrame({"t": t, "temp": h["temperature_2m"], "wind": h["wind_speed_10m"], "pr": h["precipitation"]}).set_index("t")
            for x in grp.itertuples():
                k = x.kick.floor("h")
                w = df.loc[k:k + pd.Timedelta(hours=3)]
                if len(w):
                    out.append((x.game_id, 0, float(w["temp"].mean()), float(w["wind"].mean()), float(w["pr"].sum())))
        return out

    from concurrent.futures import ThreadPoolExecutor
    done = 0
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(jobs), 40):
            if time.time() - t0 > MAX_MIN * 60:
                log("time limit reached; run again to continue")
                break
            for res in ex.map(work, jobs[i:i + 40]):
                rows += res
            done = min(i + 40, len(jobs))
            log(f"  {done}/{len(jobs)} requests, {len(rows)} games so far ({time.time() - t0:.0f}s)")
            pd.concat([have, pd.DataFrame(rows, columns=have.columns)]).drop_duplicates("game_id").to_csv(OUT, index=False)
    out = pd.concat([have, pd.DataFrame(rows, columns=have.columns)]).drop_duplicates("game_id")
    out.to_csv(OUT, index=False)
    log(f"saved {len(out)} games with weather ({int(out['indoor'].sum())} indoor) to {OUT}")
    return 0 if len(out) else 1


if __name__ == "__main__":
    sys.exit(main())
