"""Returning production for every FBS team and season (CollegeFootballData.com /player/returning) into data/cfb_returning.csv.
Columns: season, team, pct_ppa (share of last season's production that returns), pct_pass, pct_rec, pct_rush, usage.
Needs CFBD_API_KEY, same as cfb_fetch.py."""
import os
import sys

import pandas as pd

import cfb_fetch as F

OUT = os.path.join(F.HERE, "data", "cfb_returning.csv")


def main():
    if not F.KEY:
        F.log("CFBD_API_KEY is not set.")
        return 1
    rows = []
    for y in range(F.FIRST, F.current_season() + 1):
        try:
            data = F.get("/player/returning", year=y)
        except SystemExit as e:
            F.log(f"  {y}: skipped ({str(e)[:120]})")
            continue
        n = 0
        for x in data or []:
            if not isinstance(x, dict) or not x.get("team"):
                continue
            rows.append({"season": y, "team": x["team"], "pct_ppa": F.pick(x, "percentPPA", "percent_ppa"),
                         "pct_pass": F.pick(x, "percentPassingPPA", "percent_passing_ppa"),
                         "pct_rec": F.pick(x, "percentReceivingPPA", "percent_receiving_ppa"),
                         "pct_rush": F.pick(x, "percentRushingPPA", "percent_rushing_ppa"), "usage": F.pick(x, "usage")})
            n += 1
        F.log(f"  {y}: {n} teams")
    df = pd.DataFrame(rows)
    if df.empty:
        F.log("nothing returned")
        return 1
    df.to_csv(OUT, index=False)
    F.log(f"saved {len(df)} rows to {OUT}; pct_ppa average {pd.to_numeric(df['pct_ppa'], errors='coerce').mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
