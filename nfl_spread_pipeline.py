# NFL spread-model pipeline (consolidated): data -> ratings -> unit model -> weekly unit board
# One script. Works from a FRESH Colab session. Every slow stage is cached on Drive, so a
# disconnect only costs a quick re-read. Caches older than MAX_AGE_DAYS rebuild automatically,
# so running it weekly refreshes the data. Output: unit board + team unit ratings (CSV on Drive).
#
# Needs (optional): MyDrive/nfl_odds/game_lines.csv for the 2023-25 out-of-sample check vs the market.

import os
import re
import sys
import time
import subprocess
import numpy as np
import pandas as pd

try:
    import nflreadpy as nfl
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nflreadpy"], check=False)
    import nflreadpy as nfl
from sklearn.linear_model import Ridge
try:                                     # Colab: store everything on Google Drive
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except Exception:
    IN_COLAB = False                     # GitHub Actions / local machine: store inside the repo folder

# ================================================================== settings
if IN_COLAB:
    BASE = "/content/drive/MyDrive"
    ODDS_DIR = f"{BASE}/nfl_odds"
    OUT = f"{BASE}/nfl_spread_model"
    BOARD_DIR = OUT
else:
    ROOT = os.environ.get("NFL_ROOT", os.getcwd())
    ODDS_DIR = OUT = f"{ROOT}/data"        # put game_lines.csv in data/ for the market comparison
    BOARD_DIR = f"{ROOT}/boards"           # dated board files, committed to the repo
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")   # optional free-tier key: captures the current market spreads
FIRST_SEASON, LAST_SEASON, PBP_FROM = 2020, 2026, 2019
M, LAM, K_QB = 12, 0.5, 150          # prior = M games of last season's average x LAM; QB shrinkage (dropbacks)
TEST_SEASONS = [2023, 2024, 2025]
QB_REPLACEMENT_PRIOR = True          # True = a QB with little tape starts near what a new QB usually produces, not near average
FORCE_REBUILD = False                # True = ignore every cache
MAX_AGE_DAYS = 3                     # caches older than this are rebuilt
MARKET_CSV = None                    # optional: CSV with game_id, market_home_spread (positive = home favored)
SIGN_CONSTRAINED = True              # True = every weight must be >= 0 (a stat can only help the team it favors)
os.makedirs(OUT, exist_ok=True)
os.makedirs(BOARD_DIR, exist_ok=True)

T0 = time.time()
def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}")

def fresh(path):
    return (not FORCE_REBUILD) and os.path.exists(path) and (time.time() - os.path.getmtime(path)) < MAX_AGE_DAYS * 86400

def load_years(fn, years, label, **kw):
    parts = []
    for s in years:
        try:
            parts.append(fn(seasons=[s], **kw).to_pandas())
        except Exception as e:
            print(f"    {label} {s}: not loaded ({str(e)[:60]})")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

_PBP = None
def get_pbp():
    """Play-by-play loaded once and shared by every stage that needs it."""
    global _PBP
    if _PBP is None:
        log("loading play-by-play (a minute or two) ...")
        _PBP = (nfl.load_pbp(seasons=list(range(PBP_FROM, LAST_SEASON + 1)))
                   .select(["game_id", "posteam", "defteam", "play_type", "epa", "success", "wp", "sack",
                            "qb_dropback", "qb_epa", "passer_player_id", "passer_player_name"]).to_pandas())
    return _PBP

# ================================================================== stage 1: games + team-games
def build_games_and_team_games():
    sched = nfl.load_schedules(seasons=list(range(PBP_FROM, LAST_SEASON + 1))).to_pandas()
    keep = ["game_id", "season", "week", "game_type", "gameday", "gametime", "home_team", "away_team",
            "location", "div_game", "roof", "surface", "temp", "wind", "home_rest", "away_rest",
            "spread_line", "total_line", "home_score", "away_score"]
    sched = sched[[c for c in keep if c in sched.columns]].copy()
    sched["kickoff_utc"] = (
        pd.to_datetime(sched["gameday"] + " " + sched["gametime"].fillna("13:00"), format="%Y-%m-%d %H:%M")
          .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
          .dt.tz_convert("UTC"))
    g = sched[sched["season"] >= FIRST_SEASON].copy()
    g["is_playoff"] = (g["game_type"] != "REG").astype(int)
    g["neutral_site"] = (g["location"] == "Neutral").astype(int)
    g = g.rename(columns={"temp": "temp_actual", "wind": "wind_actual",
                          "spread_line": "mkt_close_nfl", "total_line": "total_close_nfl"})
    g["home_margin"] = g["home_score"] - g["away_score"]
    g["total_pts"] = g["home_score"] + g["away_score"]
    gl_path = f"{ODDS_DIR}/game_lines.csv"
    lcols = ["spread_open_wk", "spread_close", "total_open_wk", "total_close"]
    if os.path.exists(gl_path):
        g = g.merge(pd.read_csv(gl_path)[["game_id"] + lcols], on="game_id", how="left")
    else:
        print("    (no game_lines.csv found: the out-of-sample check will use the closing spread from the NFL schedule)")
        for c in lcols:
            g[c] = np.nan
    g["mkt_open"] = -g["spread_open_wk"]
    g["mkt_close"] = -g["spread_close"]
    if not os.path.exists(gl_path):                          # no purchased lines: use nflverse's closing spread (home favored = +)
        g["mkt_close"] = g["mkt_close_nfl"]
        g["mkt_open"] = g["mkt_close_nfl"]
    g["mkt_total_open"], g["mkt_total_close"] = g["total_open_wk"], g["total_close"]
    g = g.drop(columns=lcols).sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)

    pbp = get_pbp()
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna() & pbp["posteam"].notna()].copy()
    p["is_pass"] = (p["play_type"] == "pass").astype(int)
    comp = p[p["wp"].between(0.05, 0.95)]
    off = (p.groupby(["game_id", "posteam"])
             .agg(off_epa=("epa", "mean"), off_success=("success", "mean"),
                  plays=("epa", "size"), pass_rate=("is_pass", "mean"))
             .reset_index().rename(columns={"posteam": "team"}))
    off_c = comp.groupby(["game_id", "posteam"]).agg(off_epa_comp=("epa", "mean")).reset_index().rename(columns={"posteam": "team"})
    dfn = (p.groupby(["game_id", "defteam"]).agg(def_epa=("epa", "mean"), def_success=("success", "mean"))
             .reset_index().rename(columns={"defteam": "team"}))
    dfn_c = comp.groupby(["game_id", "defteam"]).agg(def_epa_comp=("epa", "mean")).reset_index().rename(columns={"defteam": "team"})
    tgm = (off.merge(off_c, on=["game_id", "team"], how="left").merge(dfn, on=["game_id", "team"], how="left")
              .merge(dfn_c, on=["game_id", "team"], how="left"))
    info = sched[["game_id", "season", "week", "game_type", "kickoff_utc", "home_team", "away_team",
                  "home_score", "away_score"]]
    tgm = tgm.merge(info, on="game_id", how="inner")
    tgm["is_home"] = (tgm["team"] == tgm["home_team"]).astype(int)
    tgm["opp"] = np.where(tgm["is_home"] == 1, tgm["away_team"], tgm["home_team"])
    tgm["points_for"] = np.where(tgm["is_home"] == 1, tgm["home_score"], tgm["away_score"])
    tgm["points_against"] = np.where(tgm["is_home"] == 1, tgm["away_score"], tgm["home_score"])
    tgm = (tgm.drop(columns=["home_team", "away_team", "home_score", "away_score"])
              .sort_values(["team", "kickoff_utc"]).reset_index(drop=True))
    return g, tgm

gp, tp = f"{OUT}/games.csv", f"{OUT}/team_games.csv"
if fresh(gp) and fresh(tp):
    log("cached: games.csv, team_games.csv")
    games, team_games = pd.read_csv(gp), pd.read_csv(tp)
else:
    log("building games.csv, team_games.csv ...")
    games, team_games = build_games_and_team_games()
    games.to_csv(gp, index=False)
    team_games.to_csv(tp, index=False)
games["kickoff_utc"] = pd.to_datetime(games["kickoff_utc"], utc=True)
team_games["kickoff_utc"] = pd.to_datetime(team_games["kickoff_utc"], utc=True)

# the regular-season team-game table every other stage attaches columns to
tg = team_games[team_games["game_type"] == "REG"].sort_values(["team", "kickoff_utc"]).reset_index(drop=True)
GROUPS = tg.groupby(["team", "season"]).indices           # row positions per team-season, in time order
N_TG = len(tg)

def add_cols(new, on):
    """Left-merge new columns into tg without changing row order (GROUPS stays valid)."""
    global tg
    tg = tg.drop(columns=[c for c in new.columns if c not in on and c in tg.columns]).merge(new, on=on, how="left")
    assert len(tg) == N_TG, "merge changed the number of rows"

def prev_sums(col, decay=1.0):
    """Sum and count of this team's PREVIOUS games this season (current game excluded)."""
    vals, S, N = tg[col].values, np.zeros(len(tg)), np.zeros(len(tg))
    for idx in GROUPS.values():
        acc = nef = 0.0
        for i in idx:
            S[i], N[i] = acc, nef
            acc, nef = decay * acc + vals[i], decay * nef + 1.0
    return S, N

# ================================================================== stage 2: pass/rush/sack components
def build_split():
    pbp = get_pbp()
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna() & pbp["posteam"].notna()
            & pbp["wp"].between(0.05, 0.95)].copy()
    p["sack"] = p["sack"].fillna(0.0)
    ps, rs = p[p["play_type"] == "pass"], p[p["play_type"] == "run"]
    parts = [
        ps.groupby(["game_id", "posteam"]).agg(off_pass=("epa", "mean"), off_sack=("sack", "mean")).reset_index().rename(columns={"posteam": "team"}),
        rs.groupby(["game_id", "posteam"]).agg(off_rush=("epa", "mean")).reset_index().rename(columns={"posteam": "team"}),
        ps.groupby(["game_id", "defteam"]).agg(def_pass=("epa", "mean"), def_sack=("sack", "mean")).reset_index().rename(columns={"defteam": "team"}),
        rs.groupby(["game_id", "defteam"]).agg(def_rush=("epa", "mean")).reset_index().rename(columns={"defteam": "team"}),
    ]
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["game_id", "team"], how="outer")
    return out

sp = f"{OUT}/team_games_split.csv"
if fresh(sp):
    log("cached: team_games_split.csv")
    split = pd.read_csv(sp)
else:
    log("building team_games_split.csv ...")
    split = build_split()
    split.to_csv(sp, index=False)
add_cols(split, ["game_id", "team"])

# ================================================================== stage 3: PFR pressure + drops, Next Gen Stats
def build_pfr():
    pas = load_years(nfl.load_pfr_advstats, range(2018, LAST_SEASON + 1), "PFR pass", stat_type="pass", summary_level="week")
    dfn = load_years(nfl.load_pfr_advstats, range(2018, LAST_SEASON + 1), "PFR def", stat_type="def", summary_level="week")
    for df in (pas, dfn):
        if "game_type" in df:
            df.drop(df[df["game_type"] != "REG"].index, inplace=True)
    a = pas.groupby(["game_id", "team"], as_index=False)[["times_pressured", "passing_drops"]].sum().rename(
        columns={"times_pressured": "press_taken"})
    b = dfn.groupby(["game_id", "team"], as_index=False)["def_pressures"].sum().rename(
        columns={"def_pressures": "press_made"})
    return a.merge(b, on=["game_id", "team"], how="outer")

def build_ngs():
    out = None
    for stat_type, spec, wcol in [
            ("rushing", {"rush_ryoe": "rush_yards_over_expected_per_att"}, "rush_attempts"),
            ("receiving", {"rec_sep": "avg_separation", "rec_yac": "avg_yac_above_expectation"}, "targets")]:
        d = load_years(nfl.load_nextgen_stats, range(2019, LAST_SEASON + 1), f"NGS {stat_type}", stat_type=stat_type)
        if "season_type" in d:
            d = d[d["season_type"] == "REG"]
        d = d[d["week"] > 0].dropna(subset=[wcol])
        res = {}
        for name, v in spec.items():
            dd = d.dropna(subset=[v]).assign(_vw=lambda x: x[v] * x[wcol])
            s = dd.groupby(["season", "week", "team_abbr"]).agg(vw=("_vw", "sum"), w=(wcol, "sum"))
            res[name] = s["vw"] / s["w"]
        part = pd.DataFrame(res).reset_index().rename(columns={"team_abbr": "team"})
        part[["season", "week"]] = part[["season", "week"]].astype(int)
        out = part if out is None else out.merge(part, on=["season", "week", "team"], how="outer")
    return out

pp, npth = f"{OUT}/pfr_team_games.csv", f"{OUT}/ngs_team_weeks.csv"
if fresh(pp):
    log("cached: pfr_team_games.csv")
    pfr = pd.read_csv(pp)
else:
    log("building PFR pressure table ...")
    pfr = build_pfr()
    pfr.to_csv(pp, index=False)
if fresh(npth):
    log("cached: ngs_team_weeks.csv")
    ngs = pd.read_csv(npth)
else:
    log("building Next Gen Stats table ...")
    ngs = build_ngs()
    ngs.to_csv(npth, index=False)
add_cols(pfr, ["game_id", "team"])
add_cols(ngs, ["season", "week", "team"])

tg["dropbacks"] = tg["plays"] * tg["pass_rate"]
db = pd.Series(tg["dropbacks"].values, index=pd.MultiIndex.from_arrays([tg["game_id"], tg["team"]]))
opp_db = db.reindex(pd.MultiIndex.from_arrays([tg["game_id"], tg["opp"]])).values
tg["press_allowed"] = (tg["press_taken"] / tg["dropbacks"]).clip(0, 1)
tg["press_gen"] = (tg["press_made"] / opp_db).clip(0, 1)
tg["drop_rate"] = tg["passing_drops"] / tg["dropbacks"].replace(0, np.nan)
FEAT_COLS = ["press_allowed", "press_gen", "off_sack", "def_sack", "def_rush", "def_pass",
             "rush_ryoe", "rec_sep", "rec_yac", "drop_rate"]
print("    data coverage, share of team-games filled (2023+):",
      {c: round(float(tg.loc[tg["season"] >= 2023, c].notna().mean()), 2) for c in FEAT_COLS})

# ================================================================== stage 4: quarterbacks
def build_qb():
    pbp = get_pbp()
    p = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna() & pbp["qb_epa"].notna() & pbp["posteam"].notna()]
    return (p.groupby(["game_id", "posteam", "passer_player_id"])
             .agg(name=("passer_player_name", "first"), db=("qb_epa", "size"), epa_sum=("qb_epa", "sum"))
             .reset_index())

qp = f"{OUT}/qb_games.csv"
if fresh(qp):
    log("cached: qb_games.csv")
    q0 = pd.read_csv(qp)
else:
    log("building qb_games.csv ...")
    q0 = build_qb()
    q0.to_csv(qp, index=False)
q0 = q0.merge(tg[["game_id", "kickoff_utc"]].drop_duplicates("game_id"), on="game_id", how="inner")

def qb_tables(K):
    """QB rating = EPA/dropback from earlier games, shrunk to 0 with K dropbacks.
       known[(game, team)] = rating of the team's PREVIOUS starter (what you know before kickoff);
       latest = each team's most recent starter right now."""
    q = q0.sort_values(["passer_player_id", "kickoff_utc"]).reset_index(drop=True)
    g = q.groupby("passer_player_id")
    q["post"] = (g["epa_sum"].cumsum() + K * QB_PRIOR) / (g["db"].cumsum() + K)
    st = (q.sort_values(["game_id", "posteam", "db"], ascending=[True, True, False])
            .drop_duplicates(["game_id", "posteam"]).sort_values(["posteam", "kickoff_utc"]))
    cg = q.groupby("passer_player_id")
    q["pre"] = ((cg["epa_sum"].cumsum() - q["epa_sum"]) + K * QB_PRIOR) / ((cg["db"].cumsum() - q["db"]) + K)
    st = st.merge(q[["game_id", "posteam", "passer_player_id", "pre"]], on=["game_id", "posteam", "passer_player_id"], how="left")
    st["prev_post"] = st.groupby("posteam")["post"].shift(1)
    prev_id = st.groupby("posteam")["passer_player_id"].shift(1)
    global QB_CHG                                          # True when the starter differs from the team's previous starter
    QB_CHG = pd.Series((prev_id.notna() & (prev_id != st["passer_player_id"])).values,
                       index=pd.MultiIndex.from_arrays([st["game_id"], st["posteam"]]))
    known = pd.Series(st["prev_post"].fillna(QB_PRIOR).values,
                      index=pd.MultiIndex.from_arrays([st["game_id"], st["posteam"]]))
    last = st.groupby("posteam").tail(1)
    actual = pd.Series(st["pre"].values, index=pd.MultiIndex.from_arrays([st["game_id"], st["posteam"]]))
    return (known, pd.Series(last["post"].values, index=last["posteam"].values),
            pd.Series(last["name"].values, index=last["posteam"].values), actual)

# prior for a QB with no track record: how every QB did in his first two games (survivors and busts alike)
_first = q0.sort_values("kickoff_utc").groupby("passer_player_id").head(2)
QB_PRIOR = float(_first["epa_sum"].sum() / _first["db"].sum()) if QB_REPLACEMENT_PRIOR else 0.0
log(f"QB prior (rating for a QB with no record): {QB_PRIOR:+.3f} EPA/dropback")
known, qb_post, qb_name, qb_actual = qb_tables(K_QB)
# rating of the QB who actually played, known before kickoff; team stats below embed whoever played earlier
tg["qb_pre"] = qb_actual.reindex(pd.MultiIndex.from_arrays([tg["game_id"], tg["team"]])).fillna(QB_PRIOR).values

# ================================================================== stage 5: injuries (burden = status weight x recent snap share)
def build_injury_burden():
    seasons = range(PBP_FROM, LAST_SEASON + 1)
    inj = load_years(nfl.load_injuries, seasons, "injuries")
    snap = load_years(nfl.load_snap_counts, seasons, "snap counts")
    for df in (inj, snap):
        df.dropna(subset=["season", "week"], inplace=True)
        df["season"], df["week"] = df["season"].astype(int), df["week"].astype(int)

    def norm(s):
        s = re.sub(r"[^a-z ]", "", str(s).lower())
        return re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)).strip()

    if "game_type" in snap:
        snap = snap[snap["game_type"] == "REG"]
    snap = snap.copy()
    snap["name"] = snap["player"].map(norm)
    snap["share"] = snap[["offense_pct", "defense_pct"]].max(axis=1).fillna(0.0)
    snap["order"] = snap["season"] * 100 + snap["week"]
    snap = snap.sort_values("order")
    snap["roll3"] = snap.groupby(["name", "team"])["share"].transform(lambda s: s.rolling(3, min_periods=1).mean())

    if "game_type" in inj:
        inj = inj[inj["game_type"] == "REG"]
    inj = inj.copy()
    inj["name"] = inj["full_name"].map(norm)
    inj["order"] = inj["season"] * 100 + inj["week"]
    inj["w"] = inj["report_status"].map({"Out": 1.0, "Doubtful": 0.85, "Questionable": 0.25}).fillna(0.0)
    inj = inj[inj["w"] > 0].sort_values("order")
    mm = pd.merge_asof(inj, snap[["order", "name", "team", "roll3"]].sort_values("order"),
                       on="order", by=["name", "team"], direction="backward", allow_exact_matches=False)
    print(f"    injured players matched to prior snap data: {mm['roll3'].notna().mean():.0%}")
    group = {"QB": "QB", "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
             "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
             "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB", "WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB"}
    mm["grp"] = mm["position"].map(group) if "position" in mm else np.nan
    mm = mm.dropna(subset=["grp"])
    mm["val"] = mm["w"] * mm["roll3"].fillna(0.0)
    return mm.groupby(["season", "week", "team", "grp"])["val"].sum().unstack(fill_value=0).reset_index()

ip = f"{OUT}/injury_burden.csv"
if fresh(ip):
    log("cached: injury_burden.csv")
    bur = pd.read_csv(ip)
else:
    log("building injury_burden.csv ...")
    bur = build_injury_burden()
    bur.to_csv(ip, index=False)
GR = [c for c in bur.columns if c not in ("season", "week", "team")]

# ================================================================== stage 6: unit model
# feature, unit, team-stat column in tg, sign (+1 = higher is better for the team)
SPEC = [("qb_rating", "QB", None, +1), ("qb_swap", "QB", "qb_pre", +1),
        ("prot_press", "O-line", "press_allowed", -1), ("prot_sack", "O-line", "off_sack", -1),
        ("run_block", "O-line", "rush_ryoe", +1),
        ("sep", "Skill", "rec_sep", +1), ("yac", "Skill", "rec_yac", +1), ("drops", "Skill", "drop_rate", -1),
        ("press_gen", "Front", "press_gen", +1), ("sack_gen", "Front", "def_sack", +1),
        ("run_def", "Front", "def_rush", -1),
        ("pass_def", "Coverage", "def_pass", -1)]
INJ = [("qb_inj", "QB", ["QB"]), ("ol_inj", "O-line", ["OL"]), ("sk_inj", "Skill", ["WR", "TE", "RB"]),
       ("fr_inj", "Front", ["DL", "LB"]), ("cov_inj", "Coverage", ["DB"])]
UNITS = ["QB", "O-line", "Skill", "Front", "Coverage"]
CTX = ["home_flag", "rest_diff"]
unit_of = {n: u for n, u, *_ in SPEC}
unit_of.update({n: u for n, u, _ in INJ})
FEATS_U = [s[0] for s in SPEC] + [i[0] for i in INJ]
ALL = FEATS_U + CTX

key = pd.MultiIndex.from_arrays([tg["game_id"], tg["team"]])
def rating_hist(col):
    """Pregame rating of every team before every game: this season's earlier games + last season's average."""
    z = (tg[col] - tg[col].mean()).fillna(0.0)
    tg["_z"] = z
    pm = tg.groupby(["team", "season"])["_z"].mean().reset_index()
    pm["season"] += 1
    prior = tg[["team", "season"]].merge(pm, on=["team", "season"], how="left")["_z"].fillna(0.0).values
    S, N = prev_sums("_z", 1.0)
    return pd.Series((S + M * LAM * prior) / (N + M), index=key)

log("building the training frame ...")
F = games[games["is_playoff"] == 0].copy().reset_index(drop=True)
mi = lambda c: pd.MultiIndex.from_arrays([F["game_id"], F[c]])
F["home_flag"] = 1 - F["neutral_site"]
F["rest_diff"] = (F["home_rest"] - F["away_rest"]).clip(-7, 7).fillna(0.0)
for name, unit, col, sign in SPEC:
    if col is None:                        # level: the QB who actually played (known before kickoff)
        F[name] = np.nan_to_num(qb_actual.reindex(mi("home_team")).values - qb_actual.reindex(mi("away_team")).values)
    elif name == "qb_swap":                # swap: today's QB minus the QB the team's stats were built with
        R, mu = rating_hist("qb_pre"), tg["qb_pre"].mean()
        sw = lambda c: qb_actual.reindex(mi(c)).values - (R.reindex(mi(c)).values + mu)
        F[name] = np.nan_to_num(sw("home_team") - sw("away_team"))
    else:
        R = rating_hist(col)
        F[name] = sign * (R.reindex(mi("home_team")).values - R.reindex(mi("away_team")).values)
for side in ("home", "away"):
    bb = bur.rename(columns={g: f"{side}_{g}" for g in GR}).rename(columns={"team": f"{side}_team"})
    F = F.merge(bb, on=["season", "week", f"{side}_team"], how="left")
for name, unit, groups in INJ:                                   # + = home LESS hurt than away
    F[name] = -sum(F[f"home_{g}"].fillna(0.0) - F[f"away_{g}"].fillna(0.0) for g in groups if f"home_{g}" in F)
F[ALL] = F[ALL].fillna(0.0)
F["qb_chg"] = (QB_CHG.reindex(mi("home_team")).fillna(False).astype(bool).values
               | QB_CHG.reindex(mi("away_team")).fillna(False).astype(bool).values)

def fit(df, feats, alpha=20, target="home_margin"):
    X, y = df[feats].values, df[target].values
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    mdl = Ridge(alpha=alpha, fit_intercept=False, positive=SIGN_CONSTRAINED).fit(X / sd, y)
    return pd.Series(mdl.coef_ / sd, index=feats)

# ---- honest out-of-sample check against the market
ok = set(F[F["season"].isin(TEST_SEASONS) & F["home_margin"].notna() & F["mkt_open"].notna()
           & F["mkt_close"].notna()]["game_id"])
rows = []
for s in TEST_SEASONS:
    tr = F[(F["season"] < s) & F["home_margin"].notna()]
    te = F[F["season"] == s].copy()
    te["pred"] = te[ALL].values @ fit(tr, ALL).values
    _noswap = [c for c in ALL if c != "qb_swap"]
    te["pred0"] = te[_noswap].values @ fit(tr, _noswap).values
    rows.append(te)
r = pd.concat(rows)
r = r[r["game_id"].isin(ok) & r["home_margin"].notna()]
mae = lambda p, y: float(np.abs(p - y).mean())
if len(r):
    print(f"\nOUT-OF-SAMPLE {TEST_SEASONS} [sign-constrained: {SIGN_CONSTRAINED}], {len(r)} identical games: unit model MAE "
          f"{mae(r['pred'], r['home_margin']):.3f} | closing market {mae(r['mkt_close'], r['home_margin']):.3f} "
          f"(expect about 10.2-10.3 vs 9.74)")
    _c = r[r["qb_chg"]]
    if len(_c):
        print(f"  games where a team changed starters ({len(_c)}): with qb_swap {mae(_c['pred'], _c['home_margin']):.3f} | "
              f"without {mae(_c['pred0'], _c['home_margin']):.3f} | closing market {mae(_c['mkt_close'], _c['home_margin']):.3f}")
    print(f"  all games: with qb_swap {mae(r['pred'], r['home_margin']):.3f} | without {mae(r['pred0'], r['home_margin']):.3f}")
else:
    print("\n(no market lines available: skipped the out-of-sample check)")

# ---- final weights on every completed game
done = F[F["home_margin"].notna()]
coef = fit(done, ALL)
print(f"\nFINAL WEIGHTS fit on {len(done)} games [sign-constrained: {SIGN_CONSTRAINED}] "
      f"(points of margin per 1 unit of the home-minus-away difference):")
for u in UNITS:
    feats = [f for f in FEATS_U if unit_of[f] == u]
    swing = float((done[feats].values @ coef[feats].values).std())
    print(f"  {u:<9} typical swing {swing:.2f} pts | " + ", ".join(f"{f} {coef[f]:+.2f}" for f in feats))
print(f"  context   home_flag {coef['home_flag']:+.2f}, rest_diff {coef['rest_diff']:+.2f}")

# ---- what weights would explain the MARKET's line? (same features, same constraints, target = closing spread)
_mk = F[F["mkt_close"].notna() & F["home_margin"].notna()]
cm = None
if len(_mk) > 300:
    cm = fit(_mk, ALL, target="mkt_close")
    _resid = _mk["mkt_close"] - _mk[ALL].values @ cm.values
    print(f"\nHow much of the market line our data can reproduce: typical miss {float(_resid.abs().mean()):.2f} pts "
          f"(std {float(_resid.std()):.2f}); the market line itself has std {float(_mk['mkt_close'].std()):.2f}")
    print(f"\nMARKET-IMPLIED WEIGHTS fit on {len(_mk)} games (target = closing spread). Typical swing in points, model weights vs market weights:")
    for u in UNITS:
        feats = [f for f in FEATS_U if unit_of[f] == u]
        a, b = (float((_mk[feats].values @ c[feats].values).std()) for c in (coef, cm))
        print(f"  {u:<9} model {a:.2f} | market {b:.2f}   " + ", ".join(f"{f} {cm[f]:+.2f}" for f in feats))
    print(f"  context   home_flag {cm['home_flag']:+.2f}, rest_diff {cm['rest_diff']:+.2f}")

# ---- diagnostic: does roster turnover explain what the market knows and our stats don't?
# turnover = share of a team's snaps (by unit) played by players who were NOT on that team last season.
# Uses whole-season snaps, so it is a test of the idea, not a pregame feature yet.
UNIT_POS = {"O-line": ["T", "G", "C", "OT", "OG", "OL"], "Skill": ["WR", "TE", "RB", "FB"],
            "Front": ["DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB", "EDGE"],
            "Coverage": ["CB", "S", "FS", "SS", "DB"]}

def roster_turnover(snap):
    pid = "pfr_player_id" if "pfr_player_id" in snap else "player"
    out = []
    for u, pos in UNIT_POS.items():
        col = "offense_snaps" if u in ("O-line", "Skill") else "defense_snaps"
        x = snap[snap["position"].isin(pos)].groupby(["season", "team", pid])[col].sum().reset_index(name="snaps")
        prev = x[["season", "team", pid]].copy()
        prev["season"] += 1
        prev["was_here"] = 1.0
        x = x.merge(prev, on=["season", "team", pid], how="left")
        x["kept"] = x["snaps"] * x["was_here"].fillna(0.0)
        a = x.groupby(["season", "team"])[["snaps", "kept"]].sum()
        a["turnover"] = 1.0 - a["kept"] / a["snaps"].replace(0, np.nan)
        a["unit"] = u
        out.append(a.reset_index()[["season", "team", "unit", "turnover"]])
    return pd.concat(out)

try:
    if cm is not None:
        _snap = load_years(nfl.load_snap_counts, range(PBP_FROM, LAST_SEASON + 1), "snap counts")
        if "game_type" in _snap:
            _snap = _snap[_snap["game_type"] == "REG"]
        tv = roster_turnover(_snap).pivot(index=["season", "team"], columns="unit", values="turnover")
        tv = tv[tv.index.get_level_values("season") >= PBP_FROM + 1]          # needs a prior season
        _mkr = _mk[_mk["season"] >= PBP_FROM + 1].copy()
        mres = _mkr["mkt_close"].values - _mkr[ALL].values @ cm.values        # market beyond our data (home +)
        gres = _mkr["home_margin"].values - _mkr[ALL].values @ coef.values    # result beyond our model (home +)
        print("\nROSTER TURNOVER TEST: home-minus-away turnover (share of snaps from new players) vs what we miss.")
        print("  corr_mkt = correlation with 'market beyond our data'; corr_res = with 'result beyond our model'; slope = points per +1.00 turnover difference")
        print(f"  {'unit':<9} {'weeks':<6} {'games':>5} {'corr_mkt':>9} {'slope_mkt':>10} {'corr_res':>9} {'slope_res':>10}")
        for u in UNIT_POS:
            h = tv[u].reindex(pd.MultiIndex.from_arrays([_mkr["season"], _mkr["home_team"]])).values
            a = tv[u].reindex(pd.MultiIndex.from_arrays([_mkr["season"], _mkr["away_team"]])).values
            d = h - a
            for label, msk in (("all", np.ones(len(d), bool)), ("1-6", (_mkr["week"] <= 6).values)):
                ok = msk & ~np.isnan(d)
                if ok.sum() < 100:
                    continue
                cmk, cr = np.corrcoef(d[ok], mres[ok])[0, 1], np.corrcoef(d[ok], gres[ok])[0, 1]
                smk, sr = np.polyfit(d[ok], mres[ok], 1)[0], np.polyfit(d[ok], gres[ok], 1)[0]
                print(f"  {u:<9} {label:<6} {ok.sum():>5} {cmk:>+9.3f} {smk:>+10.2f} {cr:>+9.3f} {sr:>+10.2f}")
        print(f"  (noise level: with ~{len(_mkr)} games a correlation under about {2/np.sqrt(len(_mkr)):.2f} is indistinguishable from zero)")
except Exception as e:
    print("Roster turnover test skipped:", str(e)[:120])

# ================================================================== stage 7: current ratings and the weekly board
TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS", "Washington Football Team": "WAS"}

def fetch_market_spreads(key):
    """Current consensus (median across books) home spread, positive = home favored. Costs 1 Odds API credit."""
    import requests
    r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
                     params={"apiKey": key, "regions": "us", "markets": "spreads", "oddsFormat": "american"},
                     timeout=60)
    r.raise_for_status()
    rows = []
    for ev in r.json():
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        pts = [oc["point"] for bk in ev.get("bookmakers", []) for mk in bk.get("markets", [])
               if mk["key"] == "spreads" for oc in mk["outcomes"]
               if oc["name"] == ev["home_team"] and -150 <= oc.get("price", -110) <= 130]
        if home and away and pts:
            rows.append({"home_team": home, "away_team": away,
                         "market_home_spread": -float(np.median(pts)), "market_books": len(pts)})
    return pd.DataFrame(rows)

season_now = int(games.loc[games["home_margin"].notna(), "season"].max())
def current_rating(col):
    z = (tg[col] - tg[col].mean()).fillna(0.0)
    now, last = tg["season"] == season_now, tg["season"] == season_now - 1
    agg = z[now].groupby(tg.loc[now, "team"]).agg(["sum", "count"])
    pr = z[last].groupby(tg.loc[last, "team"]).mean()
    return pd.Series({t: (agg["sum"].get(t, 0.0) + M * LAM * pr.get(t, 0.0)) / (agg["count"].get(t, 0.0) + M)
                      for t in sorted(set(tg["team"]))})

wk = int(games[(games["season"] == season_now) & (games["is_playoff"] == 0)
               & games["home_margin"].isna()]["week"].min())
bw = bur[(bur["season"] == season_now) & (bur["week"] == wk)].set_index("team")
print(f"\nBuilding the board for {season_now} week {wk}. Injury report found for {len(bw)} teams"
      + ("" if len(bw) else " (none yet: injury terms are zero, rerun after the Wednesday report)"))

# ---- QB grades: every starter and every seasoned backup, on the same scale the model uses
def build_qb_grades():
    q = q0.sort_values("kickoff_utc").copy()
    g = q.groupby("passer_player_id")
    q["post"] = (g["epa_sum"].cumsum() + K_QB * QB_PRIOR) / (g["db"].cumsum() + K_QB)              # the model's own rating (shrunk EPA/dropback)
    age = (q["kickoff_utc"].max() - q["kickoff_utc"]).dt.days.clip(lower=0)
    q["w"] = 0.5 ** (age / 365.0)                                              # recent games count more (1-year half-life)
    q["w_epa"], q["w_db"] = q["epa_sum"] * q["w"], q["db"] * q["w"]
    agg = q.groupby("passer_player_id").agg(
        name=("name", "last"), team=("posteam", "last"), last_game=("kickoff_utc", "max"),
        games=("game_id", "nunique"), dropbacks=("db", "sum"), epa_sum=("epa_sum", "sum"),
        w_epa=("w_epa", "sum"), w_db=("w_db", "sum"))
    agg["rating"] = q.groupby("passer_player_id")["post"].last()                # what the board uses
    agg["epa_per_db"] = agg["epa_sum"] / agg["dropbacks"]                       # raw, no shrinkage
    agg["recent_rating"] = (agg["w_epa"] + K_QB * QB_PRIOR) / (agg["w_db"] + K_QB)                  # recency-weighted, shrunk
    agg["confidence"] = agg["dropbacks"] / (agg["dropbacks"] + K_QB)            # 0..1: how much his own record counts
    starter_of = {t: n for t, n in zip(qb_name.index, qb_name.values)}
    agg["status"] = np.where(agg.apply(lambda r: starter_of.get(r["team"]) == r["name"], axis=1), "starter",
                    np.where(agg["dropbacks"] >= 300, "seasoned backup",
                    np.where(agg["dropbacks"] >= 100, "limited sample", "unproven")))
    agg = agg[(agg["last_game"] >= agg["last_game"].max() - pd.Timedelta(days=730))
              & (agg["status"] != "unproven")].copy()                           # active in the last two seasons
    ref = float(agg.loc[agg["status"] == "starter", "rating"].median())
    agg["pts_vs_avg_starter"] = (coef["qb_rating"] + coef["qb_swap"]) * (agg["rating"] - ref)      # in spread points, via the model weight
    out = agg.reset_index(drop=True)[["name", "team", "status", "pts_vs_avg_starter", "rating", "recent_rating",
                                      "epa_per_db", "dropbacks", "games", "confidence", "last_game"]]
    return out.sort_values("pts_vs_avg_starter", ascending=False).reset_index(drop=True)

qb_grades = build_qb_grades()
print("\nQB GRADES (points vs the average current starter; positive = better; top and bottom 8 starters):")
_st = qb_grades[qb_grades["status"] == "starter"]
print(pd.concat([_st.head(8), _st.tail(8)])[["name", "team", "pts_vs_avg_starter", "dropbacks"]].round(2).to_string(index=False))

# ---- manual QB overrides: qb_overrides.csv (columns: team, qb, optional week). Applies to this week's board only.
QB_SRC = {t: "last starter" for t in qb_name.index}
OVR_TEAMS = set()
_ov = os.path.join(ROOT if not IN_COLAB else OUT, "qb_overrides.csv")
if os.path.exists(_ov):
    ov = pd.read_csv(_ov)
    if "week" in ov:
        ov = ov[ov["week"].isna() | (ov["week"] == wk)]
    nm = lambda x: re.sub(r"[^a-z ]", "", str(x).lower()).strip()
    qs = q0.sort_values("kickoff_utc").copy()
    qs["key"] = qs["name"].map(nm)
    gq = qs.groupby("passer_player_id")
    qs["post"] = (gq["epa_sum"].cumsum() + K_QB * QB_PRIOR) / (gq["db"].cumsum() + K_QB)
    bykey = qs.groupby("key").tail(1).set_index("key")["post"]           # each name's latest rating
    repl = float(qb_post.quantile(0.10))                                  # replacement level: weak current starter
    for _, o in ov.iterrows():
        t, k = str(o["team"]).strip().upper(), nm(o["qb"])
        hit = bykey[bykey.index == k]
        if hit.empty:                                                     # allow just a last name
            hit = bykey[bykey.index.str.endswith(" " + k)]
        if len(hit) == 1:
            qb_post[t], src = float(hit.iloc[0]), "override (rated from his games)"
        else:
            qb_post[t], src = repl, "override (no track record: replacement level)"
        qb_name[t], QB_SRC[t] = str(o["qb"]).strip(), src
        OVR_TEAMS.add(t)
        print(f"  QB override: {t} -> {o['qb']} ({src}), rating {qb_post[t]:+.3f}")

teams = sorted(set(tg["team"]))
ADV = {}                                                     # each team's advantage per feature
for name, unit, col, sign in SPEC:
    if col is None:
        ADV[name] = qb_post - qb_post.mean()
    elif name == "qb_swap":                                  # current starter vs the QB the team's stats were built with
        _cur = current_rating("qb_pre")
        ADV[name] = qb_post.reindex(_cur.index) - (_cur + tg["qb_pre"].mean())
    else:
        ADV[name] = sign * current_rating(col)
def burden(t, groups):
    return float(sum(bw.loc[t, g] for g in groups if t in bw.index and g in bw.columns))
for name, unit, groups in INJ:
    ADV[name] = pd.Series({t: -burden(t, groups) for t in teams})
for t in OVR_TEAMS:                                          # the override already replaced the QB: no extra injury penalty
    ADV["qb_inj"][t] = 0.0

up = games[(games["season"] == season_now) & (games["week"] == wk) & (games["is_playoff"] == 0)].copy()
D = pd.DataFrame(index=up.index)
for f in FEATS_U:
    D[f] = up["home_team"].map(ADV[f]).values - up["away_team"].map(ADV[f]).values
D["home_flag"] = 1 - up["neutral_site"].values
D["rest_diff"] = (up["home_rest"] - up["away_rest"]).clip(-7, 7).fillna(0.0).values
D = D.fillna(0.0)
board = up[["game_id", "home_team", "away_team"]].copy()
board["kickoff_utc"] = up["kickoff_utc"].values
board["built_at_utc"] = pd.Timestamp.now(tz="UTC")
board["home_QB"] = up["home_team"].map(qb_name).values
board["away_QB"] = up["away_team"].map(qb_name).values
board["home_QB_src"] = up["home_team"].map(QB_SRC).values
board["away_QB_src"] = up["away_team"].map(QB_SRC).values
for u in UNITS:
    feats = [f for f in FEATS_U if unit_of[f] == u]
    board[u] = D[feats].values @ coef[feats].values
board["context"] = D[CTX].values @ coef[CTX].values
if cm is not None:                                             # same features, but weighted the way the market weights them
    board["fair_at_market_weights"] = D[ALL].values @ cm[ALL].values
board["fair_home_spread"] = board[UNITS + ["context"]].sum(axis=1)
if MARKET_CSV and os.path.exists(MARKET_CSV):
    board = board.merge(pd.read_csv(MARKET_CSV)[["game_id", "market_home_spread"]], on="game_id", how="left")
    board["market_captured_utc"] = pd.Timestamp.now(tz="UTC")
elif ODDS_API_KEY:
    try:
        mk = fetch_market_spreads(ODDS_API_KEY)
        board = board.merge(mk, on=["home_team", "away_team"], how="left")
        board["market_captured_utc"] = pd.Timestamp.now(tz="UTC")
        log(f"market spreads captured for {board['market_home_spread'].notna().sum()} of {len(board)} games")
    except Exception as e:
        print("Could not fetch market spreads:", str(e)[:100])
if "market_home_spread" in board:
    board["edge_vs_market"] = board["fair_home_spread"] - board["market_home_spread"]
    if "fair_at_market_weights" in board:                      # what our data cannot explain about the market's number
        board["market_minus_mkt_weighted"] = board["market_home_spread"] - board["fair_at_market_weights"]
board = board.sort_values("fair_home_spread", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 250)
print("\nUNIT BOARD (points; positive = favors the HOME team; units sum to the fair home spread):")
print(board.drop(columns=["game_id", "kickoff_utc", "built_at_utc", "market_captured_utc"],
                 errors="ignore").round(1).to_string(index=False))

trow = {t: {u: float(sum(coef[f] * ADV[f].get(t, 0.0) for f in FEATS_U if unit_of[f] == u)) for u in UNITS}
        for t in teams}
teamtab = pd.DataFrame(trow).T
teamtab = teamtab - teamtab.mean()
teamtab["total"] = teamtab[UNITS].sum(axis=1)
teamtab = teamtab.sort_values("total", ascending=False)
print("\nTEAM UNIT RATINGS (points vs an average team, after the latest completed games):")
print(teamtab.round(1).to_string())

stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%MZ")
board.to_csv(f"{BOARD_DIR}/unit_board_{season_now}_wk{wk}_{stamp}.csv", index=False)
board.to_csv(f"{BOARD_DIR}/unit_board_latest.csv", index=False)
teamtab.to_csv(f"{BOARD_DIR}/unit_team_ratings_latest.csv")
qb_grades.to_csv(f"{BOARD_DIR}/unit_qb_grades_latest.csv", index=False)
log(f"done. Saved the dated board, 'latest' copies and team ratings to {BOARD_DIR}")
print("\nWeekly routine: run after the Wednesday and Friday injury reports. Each dated board file is a "
      "pregame record (it carries its own build time and each game's kickoff time).")
