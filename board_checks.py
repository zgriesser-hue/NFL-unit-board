"""Sanity checks on the newest boards. Writes boards/board_checks_latest.csv (level, check, detail) and prints them.
WARN = look before you bet; INFO = good to know. The app shows these at the top of the This week and Totals tabs.
Nothing here changes a projection; it only says how far the board can be trusted right now."""
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = os.environ.get("NFL_ROOT", os.getcwd())
B = f"{ROOT}/boards"
rows = []


def add(level, check, detail):
    rows.append({"level": level, "check": check, "detail": detail})


def rd(name):
    p = f"{B}/{name}"
    return pd.read_csv(p) if os.path.exists(p) else None


now = pd.Timestamp.now(tz="UTC")
ub = rd("unit_board_latest.csv")
if ub is None or ub.empty:
    add("WARN", "Spread board", "No spread board found.")
else:
    built = pd.to_datetime(ub["built_at_utc"].iloc[0], utc=True)
    kick = pd.to_datetime(ub["kickoff_utc"], utc=True)
    age = (now - built).total_seconds() / 86400
    if age > 4.5:
        add("WARN", "Stale board", f"The spread board was built {age:.1f} days ago; rerun the workflow before betting.")
    if kick.min() < now - pd.Timedelta(hours=6) and kick.max() < now:
        add("WARN", "Old slate", "Every game on the spread board has already been played; this is last week's board.")
    n = len(ub)
    if ub["game_id"].duplicated().any():
        add("WARN", "Duplicate games", "The same game appears more than once on the spread board.")
    miss_fair = int(ub["fair_home_spread"].isna().sum())
    if miss_fair:
        add("WARN", "Missing projections", f"{miss_fair} of {n} games have no model spread.")
    if "market_home_spread" in ub:
        nm = int(ub["market_home_spread"].isna().sum())
        if nm == n:
            add("WARN", "No market lines", "No game has a market spread, so there is no edge to compare.")
        elif nm:
            games = ", ".join((ub.loc[ub["market_home_spread"].isna(), "away_team"] + "@" + ub.loc[ub["market_home_spread"].isna(), "home_team"]).head(6))
            add("WARN", "Missing lines", f"{nm} of {n} games have no market spread ({games}).")
        cap = pd.to_datetime(ub["market_captured_utc"], utc=True, errors="coerce")
        lag = (built - cap).dt.total_seconds() / 3600
        if lag.notna().any() and lag.max() > 24:
            add("WARN", "Old lines", f"Some market lines were captured {lag.max():.0f} hours before the board was built.")
    # injuries
    pool = rd("injury_player_pool_latest.csv")
    if pool is None or pool.empty:
        add("WARN", "Injuries", "No injury data on file for this board.")
    else:
        rs = pool["report_status"].fillna("").astype(str).str.strip()
        rep = pool[(rs != "") & (rs != "nan") & (rs != "Est. from practice")]
        est = pool[rs == "Est. from practice"]
        teams = set(ub["home_team"]) | set(ub["away_team"])
        with_rep = set(rep["team"]) & teams
        with_est = (set(est["team"]) & teams) - with_rep
        if with_est:
            add("WARN", "Injuries", f"Official game designations (Out/Doubtful/Questionable) are not in the data yet for {len(with_est)} of {len(teams)} teams; "
                f"their injury effect is estimated from practice participation only (rerun later for the official statuses).")
        elif not len(with_rep):
            add("WARN", "Injuries", "No team on this slate has an injury report yet, so injury effects are zero (normal on Monday; rerun after Wednesday's report).")
        elif len(with_rep) < len(teams) / 2:
            add("WARN", "Injuries", f"Only {len(with_rep)} of {len(teams)} teams on this slate have injury report entries; the report may be incomplete.")
        else:
            add("INFO", "Injuries", f"{len(with_rep)} of {len(teams)} teams on this slate have official injury report entries.")
        ov = pool[pool["override"].notna()] if "override" in pool else pool.iloc[0:0]
        if len(ov):
            add("INFO", "Manual injury overrides", "; ".join((ov["team"] + " " + ov["name"] + " " + ov["override"].astype(str)).head(8)))
    # quarterbacks
    for side in ("home", "away"):
        src = ub[f"{side}_QB_src"].fillna("")
        odd = ub[~src.str.startswith(("last starter", "override"))]
        for _, r in odd.iterrows():
            add("WARN", "Quarterback", f"{r[f'{side}_team']}: {r[f'{side}_QB']} ({r[f'{side}_QB_src']}). Confirm who starts.")
    ovq = ub[ub["home_QB_src"].fillna("").str.startswith("override") | ub["away_QB_src"].fillna("").str.startswith("override")]
    for _, r in ovq.iterrows():
        who = [f"{r[f'{s}_team']} {r[f'{s}_QB']}" for s in ("home", "away") if str(r[f"{s}_QB_src"]).startswith("override")]
        add("INFO", "QB override", "; ".join(who))
    # projection moves since the previous run for the same week
    fam = re.sub(r"_\d{8}T\d{4}Z\.csv$", "", os.path.basename(sorted(glob.glob(f"{B}/unit_board_20*_wk*_*.csv"))[-1])) if glob.glob(f"{B}/unit_board_20*_wk*_*.csv") else None
    files = sorted(glob.glob(f"{B}/{fam}_*.csv")) if fam else []
    if len(files) >= 2:
        prev = pd.read_csv(files[-2])
        m = ub.merge(prev[["game_id", "fair_home_spread"]], on="game_id", suffixes=("", "_prev"))
        m["move"] = m["fair_home_spread"] - m["fair_home_spread_prev"]
        big = m[m["move"].abs() >= 3]
        for _, r in big.iterrows():
            add("WARN", "Big projection move", f"{r['away_team']}@{r['home_team']} moved {r['move']:+.1f} since the previous run (now {r['fair_home_spread']:+.1f} home). Check injuries/QB.")
    # very large edges deserve a second look
    if "edge_vs_market" in ub:
        big = ub[ub["edge_vs_market"].abs() >= 4]
        for _, r in big.iterrows():
            add("INFO", "Large edge", f"{r['away_team']}@{r['home_team']}: model is {abs(r['edge_vs_market']):.1f} points from the market. Big gaps are usually missing news, not an edge.")

tb = rd("totals_board_latest.csv")
if tb is not None and not tb.empty:
    built = pd.to_datetime(tb["built_at_utc"].iloc[0], utc=True)
    if (now - built).total_seconds() / 86400 > 4.5:
        add("WARN", "Stale totals board", f"The totals board was built {(now - built).total_seconds() / 86400:.1f} days ago.")
    nm = int(tb["market_total"].isna().sum())
    if nm:
        add("WARN", "Missing totals", f"{nm} of {len(tb)} games have no market total.")
    if "weather_known" in tb:
        nw = tb[~tb["weather_known"].astype(bool)]
        if len(nw):
            add("INFO", "Weather", f"No forecast yet for {len(nw)} outdoor games ({', '.join((nw['away_team'] + '@' + nw['home_team']).head(6))}); treated as mild.")
    for side in ("home", "away"):
        c = f"{side}_QB_note"
        if c in tb:
            for _, r in tb[tb[c].fillna("").str.startswith("CHECK")].iterrows():
                add("WARN", "Totals quarterback", f"{r[f'{side}_team']}: {r[f'{side}_QB']} ({r[c]})")

if not any(r["level"] == "WARN" for r in rows):
    add("INFO", "All clear", "No warnings on the current boards.")
out = pd.DataFrame(rows, columns=["level", "check", "detail"])
out.insert(0, "checked_utc", now.strftime("%Y-%m-%d %H:%M"))
out.to_csv(f"{B}/board_checks_latest.csv", index=False)
for _, r in out.iterrows():
    print(f"[{r['level']}] {r['check']}: {r['detail']}")
