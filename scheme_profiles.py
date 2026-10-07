"""Defensive scheme profiles and QB-vs-scheme splits, written to boards/ for the app's Schemes tab.

Data: nflverse play-by-play + FTN charting (blitzers, 2022+) + participation (man/zone and coverage shell).
Downloads straight from the nflverse GitHub releases (cached in data/scheme_cache). Only needs pandas, numpy, requests.

Honest caveats baked into the output:
  * Scheme is a stable trait of a defense (odd/even-game correlations 0.7-0.9), so team profiles are reliable.
  * QB-specific splits are NOT reliable: in testing they did not hold up out of sample (blitz split-half r about -0.3,
    man/zone and coverage shells near zero or failing to replicate across eras). They are shown shrunk toward the
    league average with a margin of error, as context, not as a prediction input.
  * Man/zone and coverage-shell labels shift between seasons (a charting-definition change), so shares are shown
    relative to the same season's league average.
"""
import os
import numpy as np
import pandas as pd

try:
    import requests
except ImportError:                                   # pandas can read the URLs itself if requests is missing
    requests = None

HERE = os.path.dirname(os.path.abspath(__file__))
LAST_SEASON = int(os.environ.get("SCHEME_LAST_SEASON", "2026"))
WINDOW = [LAST_SEASON - 2, LAST_SEASON - 1, LAST_SEASON]          # seasons pooled for QB splits
CACHE = os.environ.get("SCHEME_CACHE", f"{HERE}/data/scheme_cache")
OUT = f"{HERE}/boards"
TAU = 0.07            # plausible true spread of QB scheme splits, EPA per dropback (optimistic)
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
os.makedirs(CACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def fetch(tag, name, ext, season, max_age_h=20):
    """Download one nflverse file to the cache (re-download for the current season when stale); None if unavailable."""
    fn = f"{CACHE}/{name}_{season}.{ext}"
    stale = (season == LAST_SEASON) and os.path.exists(fn) and (pd.Timestamp.now().timestamp() - os.path.getmtime(fn)) > max_age_h * 3600
    if os.path.exists(fn) and not stale:
        return fn
    url = f"{BASE}/{tag}/{tag}_{season}.{ext}" if name == "ftn" else f"{BASE}/{tag}/{name}_{season}.{ext}"
    try:
        if requests is None:
            raise RuntimeError("requests missing")
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            return fn if os.path.exists(fn) else None
        with open(fn, "wb") as f:
            f.write(resp.content)
        return fn
    except Exception as e:
        print(f"  could not fetch {url}: {e}")
        return fn if os.path.exists(fn) else None


def load_plays():
    cols = ["game_id", "play_id", "season", "week", "posteam", "defteam", "qb_dropback", "epa", "passer_player_id",
            "passer", "season_type"]
    seasons = sorted(set(WINDOW + [LAST_SEASON - 1, LAST_SEASON]))
    P, PART, FTN = [], [], []
    for y in seasons:
        f = fetch("pbp", "play_by_play", "csv.gz", y)
        if f:
            P.append(pd.read_csv(f, usecols=cols, low_memory=False))
        f = fetch("pbp_participation", "pbp_participation", "csv", y)
        if f:
            d = pd.read_csv(f, usecols=["nflverse_game_id", "play_id", "number_of_pass_rushers",
                                        "defense_man_zone_type", "defense_coverage_type"])
            PART.append(d.rename(columns={"nflverse_game_id": "game_id"}))
        f = fetch("ftn_charting", "ftn", "csv", y) if y >= 2022 else None
        if f:
            d = pd.read_csv(f, usecols=["nflverse_game_id", "nflverse_play_id", "n_blitzers"])
            FTN.append(d.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}))
    p = pd.concat(P)
    p = p[(p["qb_dropback"] == 1) & p["epa"].notna() & (p["season_type"] == "REG") & p["defteam"].notna()].copy()
    if PART:
        p = p.merge(pd.concat(PART), on=["game_id", "play_id"], how="left")
    else:
        p["defense_man_zone_type"] = np.nan
        p["defense_coverage_type"] = np.nan
    if FTN:
        p = p.merge(pd.concat(FTN), on=["game_id", "play_id"], how="left")
    else:
        p["n_blitzers"] = np.nan
    has = lambda c: p[c].notna()
    p["blitz"] = np.where(has("n_blitzers"), (p["n_blitzers"] >= 1).astype(float), np.nan)
    p["man"] = np.where(has("defense_man_zone_type"), (p["defense_man_zone_type"] == "MAN_COVERAGE").astype(float), np.nan)
    for k, c in (("1", "COVER_1"), ("2", "COVER_2"), ("3", "COVER_3"), ("4", "COVER_4")):
        p[f"cov{k}"] = np.where(has("defense_coverage_type"), (p["defense_coverage_type"] == c).astype(float), np.nan)
    return p


SCHEMES = {"blitz": "Blitz (vs no blitz)", "man": "Man coverage (vs zone)", "cov1": "Cover-1", "cov2": "Cover-2",
           "cov3": "Cover-3", "cov4": "Cover-4"}


def team_profiles(p):
    rows = []
    for s in (LAST_SEASON - 1, LAST_SEASON):
        d = p[p["season"] == s]
        if d.empty:
            continue
        g = d.groupby("defteam")
        t = pd.DataFrame({"dropbacks_faced": g.size()})
        for k in SCHEMES:
            t[k] = g[k].mean()
            t[k + "_n"] = g[k].count()
        t["season"] = s
        lg = {k: d[k].mean() for k in SCHEMES}
        for k in SCHEMES:
            t[k + "_league"] = lg[k]
            sd = t[k].std()
            t[k + "_z"] = (t[k] - lg[k]) / sd if sd and sd > 0 else 0.0
        rows.append(t.reset_index().rename(columns={"defteam": "team"}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(f"{OUT}/scheme_team_profiles.csv", index=False)
    print(f"wrote scheme_team_profiles.csv ({len(out)} team-seasons)")


def qb_splits(p):
    w = p[p["season"].isin(WINDOW) & p["passer_player_id"].notna()].copy()
    w["adj"] = w["epa"] - w.groupby(["season", "defteam"])["epa"].transform("mean")   # remove how good that defense is overall
    gsd = float(w["adj"].std())
    last = w.sort_values(["season", "week"]).groupby("passer_player_id").agg(
        name=("passer", "last"), team=("posteam", "last"), last_season=("season", "max"),
        dropbacks=("adj", "size"), baseline=("adj", "mean"))
    last = last[(last["last_season"] >= LAST_SEASON - 1) & (last["dropbacks"] >= 100)]
    rows = []
    for k, label in SCHEMES.items():
        q = w[w[k].notna()]
        if q.empty:
            continue
        league_diff = q.loc[q[k] == 1, "adj"].mean() - q.loc[q[k] == 0, "adj"].mean()
        g = q.groupby(["passer_player_id", k])["adj"].agg(["mean", "count"]).unstack()
        for pid in last.index.intersection(g.index):
            try:
                m1, m0 = g.loc[pid, ("mean", 1.0)], g.loc[pid, ("mean", 0.0)]
                n1, n0 = g.loc[pid, ("count", 1.0)], g.loc[pid, ("count", 0.0)]
            except KeyError:
                continue
            if not (n1 >= 15 and n0 >= 15) or np.isnan(m1) or np.isnan(m0):
                continue
            rel = (m1 - m0) - league_diff
            se = gsd * np.sqrt(1 / n1 + 1 / n0)
            rows.append({"qb_id": pid, "name": last.loc[pid, "name"], "team": last.loc[pid, "team"], "scheme": k,
                         "scheme_label": label, "n_yes": int(n1), "n_no": int(n0), "epa_yes": m1, "epa_no": m0,
                         "baseline": last.loc[pid, "baseline"], "rel_diff": rel, "se": se,
                         "shrunk": rel * TAU ** 2 / (TAU ** 2 + se ** 2), "league_diff": league_diff})
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/scheme_qb_splits.csv", index=False)
    print(f"wrote scheme_qb_splits.csv ({out['qb_id'].nunique() if len(out) else 0} QBs, {len(out)} rows)")


if __name__ == "__main__":
    plays = load_plays()
    print(f"{len(plays)} dropbacks loaded for seasons {sorted(plays['season'].unique())}")
    team_profiles(plays)
    qb_splits(plays)
