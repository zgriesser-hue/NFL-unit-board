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
try:                                     # are we inside Colab?
    import google.colab                  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False                     # GitHub Actions / local machine: store inside the repo folder
IN_COLAB = False
if ON_COLAB:                             # Colab: store everything on Google Drive, and refuse to run without it
    if not os.path.isdir("/content/drive/MyDrive"):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            print("Drive mount problem:", str(e)[:120])
    if not os.path.isdir("/content/drive/MyDrive"):
        raise SystemExit("Google Drive is not mounted, so your saved lines and caches can't be found. Run a cell with:\n"
                         "  from google.colab import drive; drive.mount('/content/drive', force_remount=True)\n"
                         "then run this again.")
    IN_COLAB = True

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
INJ_PRACTICE = os.environ.get("INJ_PRACTICE", "") == "1"   # opt-in: weight Questionable players by practice participation (set "1"); see injury_burden
CONT_USE = os.environ.get("CONT_USE", "")   # opt-in roster-continuity features, e.g. "cont_cov" or "cont_cov,cont_skill" (default: off; tested, did not help overall)
QB_REPLACEMENT_PRIOR = True          # True = a QB with little tape starts near what a new QB usually produces, not near average
FORCE_REBUILD = False                # True = ignore every cache
MAX_AGE_DAYS = 3                     # caches older than this are rebuilt
MARKET_CSV = None                    # optional: CSV with game_id, market_home_spread (positive = home favored)
SIGN_CONSTRAINED = True              # True = every weight must be >= 0 (a stat can only help the team it favors)
# ---- opt-in rating variants (defaults reproduce the current model exactly; EXPERIMENTS=1 tests them side by side)
RATING_DECAY = float(os.environ.get("RATING_DECAY", "1.0"))   # in-season game weight: 1.0 = equal; 0.9 = each older game counts 10% less
OPP_ADJ = os.environ.get("OPP_ADJ", "") == "1"                # adjust each game's stat for the opponent's pregame strength
RET_PRIOR = os.environ.get("RET_PRIOR", "") == "1"                # trust last season less for units whose roster turned over
NEW_FEATS = os.environ.get("NEW_FEATS", "")                    # comma list of: cpoe, early, xpl, succ (extra team stats), or "all"
EXTRA_FEATS = os.environ.get("EXTRA_FEATS", "") == "1"        # add offensive rush EPA (O-line) and offensive pass EPA (Skill)
BACKTEST = os.environ.get("BACKTEST", "") == "1"                  # deeper backtest of completion % over expected and success rate
EXPERIMENTS = os.environ.get("EXPERIMENTS", "") == "1"        # run the side-by-side test of the variants above (slower)
os.makedirs(OUT, exist_ok=True)
os.makedirs(BOARD_DIR, exist_ok=True)
print(f"Running in {'Colab (Google Drive)' if IN_COLAB else 'local / GitHub mode'}. Data folder: {OUT}. "
      f"game_lines.csv found: {os.path.exists(f'{ODDS_DIR}/game_lines.csv')} (looked in {ODDS_DIR})")

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
                            "qb_dropback", "qb_epa", "passer_player_id", "passer_player_name",
                            "cpoe", "down", "yards_gained"]).to_pandas())
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
    # extra candidates: completion % over expected, early-down EPA, explosive-play rate (pass 20+ yds, run 10+ yds)
    yg = p["yards_gained"].fillna(0)
    p["xpl"] = (((p["play_type"] == "pass") & (p["sack"] == 0) & (yg >= 20)) |
                ((p["play_type"] == "run") & (yg >= 10))).astype(float)
    ed = p[p["down"].isin([1, 2])]
    parts += [
        ps.groupby(["game_id", "posteam"]).agg(off_cpoe=("cpoe", "mean")).reset_index().rename(columns={"posteam": "team"}),
        ed.groupby(["game_id", "posteam"]).agg(off_early=("epa", "mean")).reset_index().rename(columns={"posteam": "team"}),
        ed.groupby(["game_id", "defteam"]).agg(def_early=("epa", "mean")).reset_index().rename(columns={"defteam": "team"}),
        p.groupby(["game_id", "posteam"]).agg(off_xpl=("xpl", "mean")).reset_index().rename(columns={"posteam": "team"}),
        p.groupby(["game_id", "defteam"]).agg(def_xpl=("xpl", "mean")).reset_index().rename(columns={"defteam": "team"}),
        ps.groupby(["game_id", "posteam"]).agg(off_psucc=("success", "mean")).reset_index().rename(columns={"posteam": "team"}),
        rs.groupby(["game_id", "posteam"]).agg(off_rsucc=("success", "mean")).reset_index().rename(columns={"posteam": "team"}),
        ps.groupby(["game_id", "defteam"]).agg(def_psucc=("success", "mean")).reset_index().rename(columns={"defteam": "team"}),
        rs.groupby(["game_id", "defteam"]).agg(def_rsucc=("success", "mean")).reset_index().rename(columns={"defteam": "team"}),
    ]
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["game_id", "team"], how="outer")
    return out

sp = f"{OUT}/team_games_split.csv"
if fresh(sp) and "def_rsucc" in pd.read_csv(sp, nrows=1).columns:
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
    if INJ_PRACTICE and "practice_status" in inj:
        # share of regular contributors who actually missed the game, 2019-23: Questionable + Full 21%, Limited 36%, Did Not Participate 58%
        pw = {"Full Participation in Practice": 0.21, "Limited Participation in Practice": 0.36,
              "Did Not Participate In Practice": 0.58}
        qm = inj["report_status"].eq("Questionable")
        inj.loc[qm, "w"] = inj.loc[qm, "practice_status"].map(pw).fillna(0.25)
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

ip = f"{OUT}/injury_burden{'_practice' if INJ_PRACTICE else ''}.csv"
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
if EXTRA_FEATS:
    SPEC += [("run_off", "O-line", "off_rush", +1), ("pass_off", "Skill", "off_pass", +1)]
# candidate team stats, grouped; each tuple is (feature, unit, team-stat column, sign)
NEW_GROUPS = {"cpoe": [("cpoe_off", "Skill", "off_cpoe", +1)],
              "early": [("early_off", "O-line", "off_early", +1), ("early_def", "Front", "def_early", -1)],
              "xpl": [("xpl_off", "Skill", "off_xpl", +1), ("xpl_def", "Coverage", "def_xpl", -1)],
              "succ": [("psucc_off", "Skill", "off_psucc", +1), ("rsucc_off", "O-line", "off_rsucc", +1),
                       ("rsucc_def", "Front", "def_rsucc", -1), ("psucc_def", "Coverage", "def_psucc", -1)]}
_want = list(NEW_GROUPS) if NEW_FEATS.strip().lower() == "all" else [x.strip() for x in NEW_FEATS.split(",") if x.strip()]
for _g in _want:
    SPEC += NEW_GROUPS[_g]
INJ = [("qb_inj", "QB", ["QB"]), ("ol_inj", "O-line", ["OL"]), ("sk_inj", "Skill", ["WR", "TE", "RB"]),
       ("fr_inj", "Front", ["DL", "LB"]), ("cov_inj", "Coverage", ["DB"])]
UNITS = ["QB", "O-line", "Skill", "Front", "Coverage"]
CTX = ["home_flag", "rest_diff"]
UNIT_POS = {"O-line": ["T", "G", "C", "OT", "OG", "OL"], "Skill": ["WR", "TE", "RB", "FB"],
            "Front": ["DE", "DT", "NT", "DL", "LB", "ILB", "OLB", "MLB", "EDGE"],
            "Coverage": ["CB", "S", "FS", "SS", "DB"]}
# roster continuity: share of the snaps played so far this season by players who were on the same team last season
CONT_DEF = [("cont_oline", "O-line", UNIT_POS["O-line"], "offense_snaps"),
            ("cont_skill", "Skill", UNIT_POS["Skill"], "offense_snaps"),
            ("cont_front", "Front", UNIT_POS["Front"], "defense_snaps"),
            ("cont_cov", "Coverage", UNIT_POS["Coverage"], "defense_snaps")]
CONT_NAMES = [c[0] for c in CONT_DEF if c[0] in CONT_USE.split(",")]
unit_of = {n: u for n, u, *_ in SPEC}
unit_of.update({n: u for n, u, _ in INJ})
unit_of.update({n: u for n, u, *_ in CONT_DEF if n in CONT_NAMES})
FEATS_U = [s[0] for s in SPEC] + [i[0] for i in INJ] + CONT_NAMES
ALL = FEATS_U + CTX

def build_continuity(snap):
    """Per (season, team, week, unit): continuity using only games BEFORE that week (cumulative)."""
    pid = "pfr_player_id" if "pfr_player_id" in snap else "player"
    sn = snap[snap["game_type"] == "REG"] if "game_type" in snap else snap
    sn = sn[["season", "week", "team", pid, "position", "offense_snaps", "defense_snaps"]].dropna(subset=["season", "week"]).copy()
    sn["season"], sn["week"] = sn["season"].astype(int), sn["week"].astype(int)
    prev = sn[["season", "team", pid]].drop_duplicates()
    prev["season"] += 1
    prev["was_here"] = 1.0
    sn = sn.merge(prev, on=["season", "team", pid], how="left")
    sn["was_here"] = sn["was_here"].fillna(0.0)
    out = {}
    for name, u, pos, col in CONT_DEF:
        x = sn[sn["position"].isin(pos)].copy()
        x["snaps"] = x[col].fillna(0.0)
        x["kept"] = x["snaps"] * x["was_here"]
        w = x.groupby(["season", "team", "week"])[["snaps", "kept"]].sum().reset_index().sort_values(["season", "team", "week"])
        w[["snaps", "kept"]] = w.groupby(["season", "team"])[["snaps", "kept"]].cumsum()
        w["cont"] = w["kept"] / w["snaps"].replace(0, np.nan)
        out[name] = w[["season", "team", "week", "cont"]].sort_values("week").reset_index(drop=True)
    return out

def cont_lookup(tab, seasons, weeks, teams):
    """Continuity as of the last game BEFORE the given week (NaN if none, e.g. week 1)."""
    left = pd.DataFrame({"season": np.asarray(seasons).astype(int), "week": np.asarray(weeks).astype(int),
                         "team": np.asarray(teams), "i": np.arange(len(teams))}).sort_values("week")
    m = pd.merge_asof(left, tab, on="week", by=["season", "team"], allow_exact_matches=False, direction="backward")
    return m.sort_values("i")["cont"].values

key = pd.MultiIndex.from_arrays([tg["game_id"], tg["team"]])
# Which opposing stat each rating is measured against: a defense's EPA allowed is expected to be higher against a
# strong offense, a line's sacks taken higher against a strong pass rush, and so on.
ADJ_PAIR = {"press_allowed": "press_gen", "press_gen": "press_allowed", "off_sack": "def_sack",
            "def_sack": "off_sack", "def_rush": "off_rush", "off_rush": "def_rush",
            "def_pass": "off_pass", "off_pass": "def_pass",
            "off_early": "def_early", "def_early": "off_early", "off_xpl": "def_xpl", "def_xpl": "off_xpl",
            "off_psucc": "def_psucc", "def_psucc": "off_psucc", "off_rsucc": "def_rsucc", "def_rsucc": "off_rsucc"}

def _prev_sums(vals, decay):
    """Weighted sum and weight of this team's PREVIOUS games this season (current game excluded)."""
    S, N = np.zeros(len(vals)), np.zeros(len(vals))
    for idx in GROUPS.values():
        acc = nef = 0.0
        for i in idx:
            S[i], N[i] = acc, nef
            acc, nef = decay * acc + vals[i], decay * nef + 1.0
    return S, N

def stat_col(col, decay, adj):
    """Column to rate: the raw team-game stat, or (adj=True) the stat minus the opponent's pregame rating of the
    matching opposing stat. Uses only games before each game, so it is safe for the backtest."""
    if not adj or col not in ADJ_PAIR:
        return col
    name = f"{col}__adj{decay:g}"
    if name not in tg.columns:
        R = rating_hist(ADJ_PAIR[col], decay, False)
        tg[name] = tg[col] - R.reindex(pd.MultiIndex.from_arrays([tg["game_id"], tg["opp"]])).fillna(0.0).values
    return name

RET = {}          # unit -> Series indexed by (season, team): relative share of last season's snaps still on the roster

def rating_hist(col, decay=None, adj=None, unit=None):
    """Pregame rating of every team before every game: this season's earlier games + last season's average.
    decay < 1 weights recent games more; adj=True is opponent-adjusted."""
    decay = RATING_DECAY if decay is None else decay
    adj = OPP_ADJ if adj is None else adj
    c = stat_col(col, decay, adj)
    z = (tg[c] - tg[c].mean()).fillna(0.0).to_numpy()
    pm = pd.DataFrame({"team": tg["team"].values, "season": tg["season"].values, "z": z}).groupby(
        ["team", "season"])["z"].mean().reset_index()
    pm["season"] += 1
    prior = tg[["team", "season"]].merge(pm, on=["team", "season"], how="left")["z"].fillna(0.0).values
    if unit is not None and unit in RET:                       # last season counts less where the roster turned over
        prior = prior * RET[unit].reindex(pd.MultiIndex.from_arrays([tg["season"], tg["team"]])).fillna(1.0).values
    S, N = _prev_sums(z, decay)
    return pd.Series((S + M * LAM * prior) / (N + M), index=key)

log("building the training frame ...")
SNAP = load_years(nfl.load_snap_counts, range(PBP_FROM, LAST_SEASON + 1), "snap counts")
CONT_TAB = build_continuity(SNAP) if CONT_NAMES else {}

def build_returning(snap):
    """Per unit and (season, team): share of LAST season's snaps (at that unit) played by players on this season's
    week-1 roster, divided by the league average that season (1.0 = typical). Known before kickoff of week 1."""
    pid = "pfr_player_id" if "pfr_player_id" in snap else "player"
    try:
        rw = nfl.load_rosters_weekly(seasons=list(range(PBP_FROM + 1, LAST_SEASON + 1))).to_pandas()
    except Exception:
        rw = nfl.load_rosters(seasons=list(range(PBP_FROM + 1, LAST_SEASON + 1))).to_pandas()
    rid = next((c for c in ("pfr_id", "pfr_player_id") if c in rw.columns), None)
    if rid is None:
        raise RuntimeError("roster table has no PFR id column")
    if "week" in rw:
        rw = rw[rw["week"] == rw.groupby("season")["week"].transform("min")]
    if "status" in rw:
        rw = rw[rw["status"].isin(["ACT", "RES", "INA", "Active", "Reserve/Injured"]) | rw["status"].isna()]
    cur = rw[["season", "team", rid]].dropna().drop_duplicates().rename(columns={rid: pid})
    sn = snap[snap["game_type"] == "REG"] if "game_type" in snap else snap
    out = {}
    for u, pos in UNIT_POS.items():
        col = "offense_snaps" if u in ("O-line", "Skill") else "defense_snaps"
        x = sn[sn["position"].isin(pos)].groupby(["season", "team", pid])[col].sum().reset_index(name="snaps")
        x["season"] += 1                                       # last season's snaps, attached to the season they feed
        x = x.merge(cur.assign(here=1.0), on=["season", "team", pid], how="left")
        x["kept"] = x["snaps"] * x["here"].fillna(0.0)
        a = x.groupby(["season", "team"])[["snaps", "kept"]].sum()
        share = a["kept"] / a["snaps"].replace(0, np.nan)
        out[u] = (share / share.groupby(level="season").transform("mean")).clip(0.4, 1.3)
    return out

try:
    RET.update(build_returning(SNAP))
    log("returning-roster shares built (" + ", ".join(f"{u}: {RET[u].std():.2f} sd" for u in RET) + ")")
except Exception as e:
    print("returning-roster shares skipped:", str(e)[:150])
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
        R = rating_hist(col, unit=unit if RET_PRIOR else None)
        F[name] = sign * (R.reindex(mi("home_team")).values - R.reindex(mi("away_team")).values)
F["qb_known"] = np.nan_to_num(known.reindex(mi("home_team")).values - known.reindex(mi("away_team")).values)
for side in ("home", "away"):
    bb = bur.rename(columns={g: f"{side}_{g}" for g in GR}).rename(columns={"team": f"{side}_team"})
    F = F.merge(bb, on=["season", "week", f"{side}_team"], how="left")
for name, unit, groups in INJ:                                   # + = home LESS hurt than away
    F[name] = -sum(F[f"home_{g}"].fillna(0.0) - F[f"away_{g}"].fillna(0.0) for g in groups if f"home_{g}" in F)
for name, unit, pos, col in CONT_DEF:                                # + = home lineup MORE continuous than away's
    if name in CONT_NAMES:
        h = cont_lookup(CONT_TAB[name], F["season"], F["week"], F["home_team"])
        a = cont_lookup(CONT_TAB[name], F["season"], F["week"], F["away_team"])
        F[name] = h - a
F[ALL] = F[ALL].fillna(0.0)
F["qb_chg"] = (QB_CHG.reindex(mi("home_team")).astype("boolean").fillna(False).to_numpy(dtype=bool)
               | QB_CHG.reindex(mi("away_team")).astype("boolean").fillna(False).to_numpy(dtype=bool))

def fit(df, feats, alpha=20, target="home_margin"):
    X, y = df[feats].values, df[target].values
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    mdl = Ridge(alpha=alpha, fit_intercept=False, positive=SIGN_CONSTRAINED).fit(X / sd, y)
    return pd.Series(mdl.coef_ / sd, index=feats)

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

# ---- honest out-of-sample check against the market
ok = set(F[F["season"].isin(TEST_SEASONS) & F["home_margin"].notna() & F["mkt_open"].notna()
           & F["mkt_close"].notna()]["game_id"])
rows = []
for s in TEST_SEASONS:
    tr = F[(F["season"] < s) & F["home_margin"].notna()]
    te = F[F["season"] == s].copy()
    _w = fit(tr, ALL)
    te["pred"] = te[ALL].values @ _w.values
    _tm = te.copy()                                              # as of Monday noon: no injury info, no starter news
    for _c in [i[0] for i in INJ]:
        _tm[_c] = 0.0
    _tm["qb_swap"] = 0.0
    _tm["qb_rating"] = _tm["qb_known"]
    te["pred_mon"] = _tm[ALL].values @ _w.values
    _th = te.copy()                                              # hybrid: injuries as finally reported, but the PREVIOUS starter at QB
    _th["qb_swap"] = 0.0
    _th["qb_rating"] = _th["qb_known"]
    te["pred_hyb"] = _th[ALL].values @ _w.values
    _noswap = [c for c in ALL if c != "qb_swap"]
    te["pred0"] = te[_noswap].values @ fit(tr, _noswap).values
    _nocont = [c for c in ALL if c not in CONT_NAMES]
    te["pred_nocont"] = te[_nocont].values @ fit(tr, _nocont).values
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
    if CONT_NAMES:
        _e = r[r["week"] <= 6]
        print(f"  roster continuity: all games with {mae(r['pred'], r['home_margin']):.3f} | without {mae(r['pred_nocont'], r['home_margin']):.3f}"
              f" ; weeks 1-6 ({len(_e)}) with {mae(_e['pred'], _e['home_margin']):.3f} | without {mae(_e['pred_nocont'], _e['home_margin']):.3f}")
    print(f"  all games: with qb_swap {mae(r['pred'], r['home_margin']):.3f} | without {mae(r['pred0'], r['home_margin']):.3f}")
else:
    print("\n(no market lines available: skipped the out-of-sample check)")

# ---- experiments: rating variants side by side (set EXPERIMENTS=1). Same games, same fit, paired by game.
def run_experiments():
    _new_names = {f[0] for g in NEW_GROUPS.values() for f in g}
    spec_base = [x for x in SPEC if x[0] not in ("run_off", "pass_off") and x[0] not in _new_names]
    extra_spec = [("run_off", "O-line", "off_rush", +1), ("pass_off", "Skill", "off_pass", +1)]
    inj_cols = [i[0] for i in INJ] + CONT_NAMES

    def frame(spec, decay, adj, ret=False):
        Fv = F.copy()
        for name, unit, col, sign in spec:
            if col is None or name == "qb_swap":
                continue                                           # QB terms do not depend on the variant
            R = rating_hist(col, decay, adj, unit if ret else None)
            Fv[name] = np.nan_to_num(sign * (R.reindex(mi("home_team")).values - R.reindex(mi("away_team")).values))
        return Fv

    def oos(Fv, feats):
        errs = []
        for sn in TEST_SEASONS:
            tr = Fv[(Fv["season"] < sn) & Fv["home_margin"].notna()]
            te = Fv[(Fv["season"] == sn) & Fv["game_id"].isin(ok)]
            errs.append(np.abs(te[feats].values @ fit(tr, feats).values - te["home_margin"].values))
        return np.concatenate(errs)

    variants = [("baseline (current model)", 1.0, False, False, []),
                ("recency decay 0.95", 0.95, False, False, []), ("recency decay 0.90", 0.90, False, False, []),
                ("recency decay 0.80", 0.80, False, False, []),
                ("opponent-adjusted", 1.0, True, False, []),
                ("opponent-adjusted + decay 0.90", 0.90, True, False, []),
                ("+ offensive rush/pass EPA", 1.0, False, True, []),
                ("opp-adjusted + offensive rush/pass EPA", 1.0, True, True, []),
                ("+ completion % over expected", 1.0, False, False, ["cpoe"]),
                ("+ early-down EPA (off and def)", 1.0, False, False, ["early"]),
                ("+ explosive-play rate (off and def)", 1.0, False, False, ["xpl"]),
                ("+ all three new stats", 1.0, False, False, ["cpoe", "early", "xpl"]),
                ("+ success rate, pass and rush (off and def)", 1.0, False, False, ["succ"]),
                ("+ all four new stat groups", 1.0, False, False, ["cpoe", "early", "xpl", "succ"]),
                ("last season trusted less after roster turnover", 1.0, False, False, [], True)]
    variants = [v if len(v) == 6 else v + (False,) for v in variants]
    if not RET:
        variants = [v for v in variants if not v[5]]
        print("  (roster-turnover variant skipped: returning-roster shares could not be built)")
    print("\nEXPERIMENTS: rating variants, out-of-sample", TEST_SEASONS, "(average miss vs actual margin; lower is better)")
    base = None
    for label, decay, adj, extra, groups, ret in variants:
        spec = spec_base + (extra_spec if extra else []) + [f for g in groups for f in NEW_GROUPS[g]]
        Fv = frame(spec, decay, adj, ret)
        feats = [x[0] for x in spec] + inj_cols + CTX
        e = oos(Fv, feats)
        if base is None:
            base = e
            print(f"  {label:<42} {e.mean():.3f}   ({len(e)} games; closing market {mae(r['mkt_close'], r['home_margin']):.3f})")
            continue
        d = base - e                                                # + = better than baseline
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        verdict = "clear gain" if d.mean() - ci > 0 else ("worse" if d.mean() + ci < 0 else "within noise")
        print(f"  {label:<42} {e.mean():.3f}   change {d.mean():+.3f} +/- {ci:.3f}  {verdict}")
    print("  (adopt a variant only when it is a clear gain; then set RATING_DECAY / OPP_ADJ / EXTRA_FEATS / NEW_FEATS / RET_PRIOR for the weekly run)")

if EXPERIMENTS and len(r):
    run_experiments()

# ---- deeper backtest of the two most promising candidates: completion % over expected and success rate.
# Season by season, early vs late weeks, share of games improved, and the weights the model actually gives them.
def run_backtest():
    base_spec = [x for x in SPEC if x[0] not in {f[0] for g in NEW_GROUPS.values() for f in g}]
    inj_cols = [i[0] for i in INJ] + CONT_NAMES
    seasons = [sn for sn in range(2022, 2026) if sn in set(F["season"])]            # 2022 trains on only 2020-21: a rough extra check
    def frame(spec):
        Fv = F.copy()
        for name, unit, col, sign in spec:
            if col is None or name == "qb_swap":
                continue
            R = rating_hist(col, unit=unit if RET_PRIOR else None)
            Fv[name] = np.nan_to_num(sign * (R.reindex(mi("home_team")).values - R.reindex(mi("away_team")).values))
        return Fv
    def run(spec):
        Fv = frame(spec)
        feats = [x[0] for x in spec] + inj_cols + CTX
        parts = []
        for sn in seasons:
            tr = Fv[(Fv["season"] < sn) & Fv["home_margin"].notna()]
            te = Fv[(Fv["season"] == sn) & Fv["home_margin"].notna()].copy()
            te["err"] = np.abs(te[feats].values @ fit(tr, feats).values - te["home_margin"].values)
            parts.append(te[["game_id", "season", "week", "err"]])
        return pd.concat(parts).reset_index(drop=True), fit(Fv[Fv["home_margin"].notna()], feats)
    cands = [("completion % over expected", ["cpoe"]), ("success rate", ["succ"]), ("both together", ["cpoe", "succ"])]
    base, _ = run(base_spec)
    print("\nBACKTEST: completion % over expected and success rate vs the current model (positive change = better)")
    print(f"  seasons tested {seasons}; {len(base)} games; each season is trained only on earlier seasons")
    for label, groups in cands:
        spec = base_spec + [f for g in groups for f in NEW_GROUPS[g]]
        cur, w = run(spec)
        d = base["err"].values - cur["err"].values
        ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
        print(f"\n  {label}: overall {cur['err'].mean():.3f} vs {base['err'].mean():.3f}   change {d.mean():+.3f} +/- {ci:.3f}"
              f"   better in {100 * (d > 0).mean():.0f}% of games")
        for sn in seasons:
            m = (cur["season"] == sn).values
            dd = d[m]
            print(f"    {sn}: {cur['err'][m].mean():.3f} vs {base['err'][m].mean():.3f}   change {dd.mean():+.3f} +/- {1.96 * dd.std(ddof=1) / np.sqrt(len(dd)):.3f}")
        for nm, m in (("weeks 1-6", (cur["week"] <= 6).values), ("weeks 7+", (cur["week"] > 6).values)):
            dd = d[m]
            print(f"    {nm:<10} change {dd.mean():+.3f} +/- {1.96 * dd.std(ddof=1) / np.sqrt(len(dd)):.3f}  ({m.sum()} games)")
        newf = [f[0] for g in groups for f in NEW_GROUPS[g]]
        print("    weights the model gives them (0 = it ignored the stat): " + ", ".join(f"{n} {w[n]:+.1f}" for n in newf))
    print("  (a real gain should show up in most seasons, not just one; a +/- band that spans zero means it could be noise)")

if BACKTEST and len(r):
    run_backtest()

# ---- the opener: Monday-noon line from game_lines.csv (only meaningful when that file is present)
if len(r) and os.path.exists(f"{ODDS_DIR}/game_lines.csv"):
    hm = r["home_margin"]
    print(f"\nVS THE OPENER ({len(r)} games). Average miss: opener {mae(r['mkt_open'], hm):.3f} | close {mae(r['mkt_close'], hm):.3f} "
          f"| unit model {mae(r['pred'], hm):.3f}")
    for label, col in (("opener", "mkt_open"), ("close", "mkt_close")):
        edge, cover = r["pred"] - r[col], hm - r[col]
        print(f"  betting the model side against the {label} (break-even at -110 is 52.4%):")
        for thr in [0, 1, 2, 3, 4]:
            sel = (edge.abs() >= thr) & (cover != 0)
            n = int(sel.sum())
            if n == 0:
                continue
            wr = float((np.sign(edge[sel]) == np.sign(cover[sel])).mean())
            print(f"    |edge| >= {thr}: {n} bets, win {wr:.1%} (+/- {np.sqrt(0.25 / n):.1%}), ROI {wr * 100 / 110 - (1 - wr):+.1%}")
    mv = r["mkt_close"] - r["mkt_open"]                          # how the line moved from open to close (home +)
    lean = r["pred"] - r["mkt_open"]                             # which way the model disagreed with the opener
    big = (mv.abs() >= 1) & (lean.abs() >= 0.5)
    if big.sum() > 30:
        agree = float((np.sign(lean[big]) == np.sign(mv[big])).mean())
        print(f"  line movement: when the line moved 1+ points, the model had leaned that way {agree:.1%} of the time "
              f"({int(big.sum())} games, +/- {np.sqrt(0.25 / big.sum()):.1%}); correlation of model-vs-open with the move: "
              f"{float(np.corrcoef(lean, mv)[0, 1]):+.3f}")
    print("  OPENER-TIMED model (what it would have said Monday noon: previous starter, no injury information):")
    print(f"    average miss: model {mae(r['pred_mon'], hm):.3f} | opener {mae(r['mkt_open'], hm):.3f}")
    edge, cover = r["pred_mon"] - r["mkt_open"], hm - r["mkt_open"]
    for thr in [0, 1, 2, 3, 4]:
        sel = (edge.abs() >= thr) & (cover != 0)
        n = int(sel.sum())
        if n:
            wr = float((np.sign(edge[sel]) == np.sign(cover[sel])).mean())
            print(f"    |edge| >= {thr}: {n} bets, win {wr:.1%} (+/- {np.sqrt(0.25 / n):.1%}), ROI {wr * 100 / 110 - (1 - wr):+.1%}")
    lean2 = r["pred_mon"] - r["mkt_open"]
    big2 = (mv.abs() >= 1) & (lean2.abs() >= 0.5)
    if big2.sum() > 30:
        print(f"    line movement: leaned the move's way {float((np.sign(lean2[big2]) == np.sign(mv[big2])).mean()):.1%} of the time "
              f"({int(big2.sum())} games); correlation {float(np.corrcoef(lean2, mv)[0, 1]):+.3f}")
    keep = ["game_id", "season", "week", "home_team", "away_team", "home_margin", "pred", "pred_mon", "mkt_open", "mkt_close"]
    r[[c for c in keep if c in r]].to_csv(f"{OUT}/oos_predictions.csv", index=False)
    log(f"saved oos_predictions.csv ({len(r)} test games) to {OUT}")

# ---- mid-week lines: Thursday-evening and Friday consensus from odds_snapshots.csv (the timing a weekly board can act on)
def build_midweek(path):
    """One row per game: median home spread (home favored = +) across books at the last snapshot on Thursday / Friday
    (ET) of game week, before kickoff. Thursday-night games have no earlier Thursday and are skipped."""
    cache = f"{OUT}/midweek_lines.csv"
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        c = pd.read_csv(cache)
        c["kick"] = pd.to_datetime(c["kick"], utc=True)
        return c
    parts = []
    for ch in pd.read_csv(path, usecols=["snapshot_ts", "event_id", "commence_time", "home_team", "away_team",
                                         "bookmaker", "market", "outcome", "point"], chunksize=2_000_000):
        ch = ch[(ch["market"] == "spreads") & (ch["outcome"] == ch["home_team"])]
        parts.append(ch.drop(columns=["market", "outcome"]))
    s = pd.concat(parts, ignore_index=True)
    s["snap"] = pd.to_datetime(s["snapshot_ts"], utc=True)
    s["kick"] = pd.to_datetime(s["commence_time"], utc=True)
    ke = s["kick"].dt.tz_convert("America/New_York")
    back = (ke.dt.weekday - 3) % 7
    thu = ke.dt.normalize() - pd.to_timedelta(back, unit="D") + pd.Timedelta(hours=23, minutes=59)
    thu = thu.where(back > 0).dt.tz_convert("UTC")
    ev = (s.drop_duplicates("event_id")[["event_id", "home_team", "away_team", "kick"]]
            .assign(home=lambda d: d["home_team"].map(TEAM_ABBR), away=lambda d: d["away_team"].map(TEAM_ABBR))
            .set_index("event_id"))
    for name, cut in (("mid_thu", thu), ("mid_fri", thu + pd.Timedelta(days=1))):
        sub = s[(s["snap"] <= cut) & (s["snap"] > cut - pd.Timedelta(hours=24)) & (s["snap"] < s["kick"] - pd.Timedelta(hours=3))]
        sub = sub[sub["snap"] == sub.groupby("event_id")["snap"].transform("max")]
        ev[name] = -sub.groupby("event_id")["point"].median()
        ev[name + "_books"] = sub.groupby("event_id")["bookmaker"].nunique()
    ev = ev.dropna(subset=["home", "away"]).reset_index(drop=True)[["home", "away", "kick", "mid_thu", "mid_fri", "mid_thu_books", "mid_fri_books"]]
    ev.to_csv(cache, index=False)
    return ev

def _bets(pred, line, hm, label):
    edge, cover = pred - line, hm - line
    out = []
    for thr in (0, 1, 2, 3):
        sel = (edge.abs() >= thr) & (cover != 0)
        n = int(sel.sum())
        if n:
            wr = float((np.sign(edge[sel]) == np.sign(cover[sel])).mean())
            out.append(f"|edge|>={thr}: {n} bets {wr:.1%} (ROI {wr * 100 / 110 - (1 - wr):+.1%})")
    print(f"    {label}: " + " | ".join(out))

_sp = f"{ODDS_DIR}/odds_snapshots.csv"
if len(r) and os.path.exists(_sp):
    log("building mid-week lines from odds_snapshots.csv (first run reads the whole file, a few minutes)")
    mw = build_midweek(_sp)
    _rr = r[["game_id", "home_team", "away_team", "kickoff_utc"]].copy()
    _rr["kickoff_utc"] = pd.to_datetime(_rr["kickoff_utc"], utc=True).astype("datetime64[ns, UTC]")
    mw["kick"] = mw["kick"].astype("datetime64[ns, UTC]")
    _m = pd.merge_asof(_rr.sort_values("kickoff_utc"), mw.sort_values("kick"), left_on="kickoff_utc", right_on="kick",
                       left_by=["home_team", "away_team"], right_by=["home", "away"],
                       tolerance=pd.Timedelta("3D"), direction="nearest")
    rm = r.merge(_m[["game_id", "mid_thu", "mid_fri", "mid_thu_books", "mid_fri_books"]], on="game_id", how="left")
    print(f"\nMID-WEEK LINES: Thursday line found for {rm['mid_thu'].notna().sum()} of {len(rm)} test games, "
          f"Friday line for {rm['mid_fri'].notna().sum()} (Thursday-night games excluded by design)")
    for tag, col in (("THURSDAY EVENING", "mid_thu"), ("FRIDAY", "mid_fri")):
        q = rm[rm[col].notna()]
        if len(q) < 40:
            print(f"  {tag}: only {len(q)} games, skipping")
            continue
        hm = q["home_margin"]
        print(f"\n  {tag} line, {len(q)} games ({q['season'].value_counts().sort_index().to_dict()}). Average miss: "
              f"that line {mae(q[col], hm):.3f} | Monday opener {mae(q['mkt_open'], hm):.3f} | close {mae(q['mkt_close'], hm):.3f}")
        print(f"    models: full information {mae(q['pred'], hm):.3f} | injuries known but previous starter {mae(q['pred_hyb'], hm):.3f} "
              f"| Monday-style (no injuries, previous starter) {mae(q['pred_mon'], hm):.3f}")
        print(f"    (full information uses final injury reports and the actual starter, which is more than anyone has on Thursday; "
              f"Monday-style uses less. Reality sits between them.)")
        for lbl, pc in (("full information", "pred"), ("injuries + previous starter", "pred_hyb"), ("Monday-style", "pred_mon")):
            _bets(q[pc], q[col], hm, f"betting {lbl} vs this line")
        mv = q["mkt_close"] - q[col]
        for lbl, pc in (("full information", "pred"), ("injuries + previous starter", "pred_hyb"), ("Monday-style", "pred_mon")):
            lean = q[pc] - q[col]
            big = (mv.abs() >= 1) & (lean.abs() >= 0.5)
            if big.sum() > 20:
                print(f"    line move from here to close of 1+ pts: {lbl} leaned that way "
                      f"{float((np.sign(lean[big]) == np.sign(mv[big])).mean()):.1%} of {int(big.sum())} games; "
                      f"correlation of lean with move {float(np.corrcoef(lean, mv)[0, 1]):+.3f}")
    rm.to_csv(f"{OUT}/oos_midweek.csv", index=False)
elif len(r):
    print(f"\n(no odds_snapshots.csv in {ODDS_DIR}: skipped the mid-week comparison)")

# ---- shrink unit weights toward the weights the market implies (market weights fit only on PRIOR seasons' closing lines)
if len(r):
    print("\nSHRINK TOWARD MARKET WEIGHTS (out of sample): blend = (1 - lam) x our weights + lam x market-implied weights, "
          "per unit; average miss vs the actual margin, lower is better.")
    groups = {"all units": UNITS + ["context"], "Skill + O-line": ["Skill", "O-line"],
              "Skill + O-line + Front": ["Skill", "O-line", "Front"], "Skill only": ["Skill"]}
    feat_unit = {f: unit_of.get(f, "context") for f in ALL}
    lams = [0, 0.25, 0.5, 0.75, 1.0]
    tab = {g: {l: [] for l in lams} for g in groups}
    for s in TEST_SEASONS:
        tr = F[(F["season"] < s) & F["home_margin"].notna()]
        trm = tr[tr["mkt_close"].notna()]
        w, wm = fit(tr, ALL), fit(trm, ALL, target="mkt_close")
        te = r[r["season"] == s]
        for g, us in groups.items():
            sel = np.array([feat_unit[f] in us for f in ALL])
            for l in lams:
                wb = w.values.copy()
                wb[sel] = (1 - l) * w.values[sel] + l * wm.values[sel]
                tab[g][l].append(np.abs(te[ALL].values @ wb - te["home_margin"].values))
    print(f"  {'blend over':<24}" + "".join(f"lam={l:<6}" for l in lams))
    for g in groups:
        print(f"  {g:<24}" + "".join(f"{np.concatenate(tab[g][l]).mean():<10.3f}" for l in lams))
    print(f"  (closing market's own miss on these games: {mae(r['mkt_close'], r['home_margin']):.3f}; "
          f"lam=1 on all units is 'our data weighted exactly like the market')")

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
        _snap = SNAP
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
def current_rating(col, decay=None, adj=None, unit=None):
    decay = RATING_DECAY if decay is None else decay
    adj = OPP_ADJ if adj is None else adj
    c = stat_col(col, decay, adj)
    z = (tg[c] - tg[c].mean()).fillna(0.0)
    now, last = tg["season"] == season_now, tg["season"] == season_now - 1
    pr = z[last].groupby(tg.loc[last, "team"]).mean()
    rf = RET[unit].xs(season_now, level="season") if unit is not None and unit in RET and season_now in RET[unit].index.get_level_values("season") else None
    out = {}
    for t in sorted(set(tg["team"])):
        v = z[now & (tg["team"] == t)].to_numpy()               # tg is in time order within each team
        w = decay ** np.arange(len(v) - 1, -1, -1)
        out[t] = ((w * v).sum() + M * LAM * pr.get(t, 0.0) * (rf.get(t, 1.0) if rf is not None else 1.0)) / (w.sum() + M)
    return pd.Series(out)

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

# ---- player pool (feeds the app's What-if tab) and optional manual injury overrides.
# injury_overrides.csv columns: team, player, status, optional week.  status = Out / Doubtful / Questionable / Healthy,
# or a number 0-1 (your own chance he misses the game).  It replaces that player's reported status for this week's board.
_norm = lambda s: re.sub(r"\s+", " ", re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", re.sub(r"[^a-z ]", "", str(s).lower()))).strip()
_GRP = {"QB": "QB", "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL", "DE": "DL", "DT": "DL",
        "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB", "CB": "DB", "S": "DB", "FS": "DB",
        "SS": "DB", "DB": "DB", "WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB"}
_STATUS_W = {"out": 1.0, "doubtful": 0.85, "questionable": 0.25, "probable": 0.0, "full": 0.0, "healthy": 0.0,
             "active": 0.0, "available": 0.0}
POOL = pd.DataFrame()
try:
    _s = SNAP.copy()
    if "game_type" in _s:
        _s = _s[_s["game_type"] == "REG"]
    _s["order"] = _s["season"].astype(int) * 100 + _s["week"].astype(int)
    _s = _s.sort_values("order")
    _s["share"] = _s[["offense_pct", "defense_pct"]].max(axis=1).fillna(0.0)
    _s["key"] = _s["player"].map(_norm)
    _s["roll3"] = _s.groupby(["key", "team"])["share"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    _last = _s.groupby(["key", "team"]).tail(1).copy()
    _last = _last[(_last["order"] >= _last["team"].map(_s.groupby("team")["order"].max()) - 3) & (_last["roll3"] >= 0.2)]
    _last["grp"] = _last["position"].map(_GRP)
    POOL = (_last.dropna(subset=["grp"])[["team", "player", "key", "position", "grp", "roll3"]]
            .rename(columns={"roll3": "snap_share", "player": "name"}).reset_index(drop=True))
    _ij = load_years(nfl.load_injuries, [season_now], "injuries")
    _ij = _ij[_ij["week"].astype(int) == wk].copy()
    if "game_type" in _ij:
        _ij = _ij[_ij["game_type"] == "REG"]
    _ij["key"] = _ij["full_name"].map(_norm)
    _ij["w"] = _ij["report_status"].map({"Out": 1.0, "Doubtful": 0.85, "Questionable": 0.25}).fillna(0.0)
    if INJ_PRACTICE and "practice_status" in _ij:
        _pw = {"Full Participation in Practice": 0.21, "Limited Participation in Practice": 0.36,
               "Did Not Participate In Practice": 0.58}
        _qm = _ij["report_status"].eq("Questionable")
        _ij.loc[_qm, "w"] = _ij.loc[_qm, "practice_status"].map(_pw).fillna(0.25)
    if "date_modified" in _ij:
        _ij = _ij.sort_values("date_modified")
    _ij = _ij.drop_duplicates(["team", "key"], keep="last")
    POOL = POOL.merge(_ij[["team", "key", "w", "report_status", "practice_status"]], on=["team", "key"], how="left")
    POOL["w_now"] = POOL.pop("w").fillna(0.0)
    POOL["report_status"] = POOL["report_status"].fillna("")
    POOL["practice_status"] = POOL["practice_status"].fillna("")
    POOL["override"] = ""
except Exception as e:
    print("Player pool for the What-if tab not built:", str(e)[:120])
    POOL = pd.DataFrame()
_io = os.path.join(ROOT if not IN_COLAB else OUT, "injury_overrides.csv")
if os.path.exists(_io) and len(POOL):
    iov = pd.read_csv(_io)
    if "week" in iov:
        iov = iov[iov["week"].isna() | (iov["week"] == wk)]
    for _, o in iov.iterrows():
        t, k = str(o["team"]).strip().upper(), _norm(o["player"])
        m = POOL[(POOL["team"] == t) & ((POOL["key"] == k) | POOL["key"].str.endswith(" " + k))]
        if len(m) != 1:
            print(f"  injury override skipped: {len(m)} matches for '{o['player']}' on {t} (use his full name; he must have played recently)")
            continue
        sv = str(o["status"]).strip().lower()
        try:
            wn = float(sv)
        except ValueError:
            wn = _STATUS_W.get(sv)
        if wn is None:
            print(f"  injury override skipped: unknown status '{o['status']}' for {o['player']}")
            continue
        i = m.index[0]
        g = POOL.loc[i, "grp"]
        if t not in bw.index:
            bw.loc[t] = 0.0
        if g not in bw.columns:
            bw[g] = 0.0
        bw.loc[t, g] = max(float(bw.loc[t, g]) + (wn - POOL.loc[i, "w_now"]) * POOL.loc[i, "snap_share"], 0.0)
        POOL.loc[i, ["w_now", "override"]] = [wn, str(o["status"])]
        print(f"  injury override: {t} {POOL.loc[i, 'name']} -> {o['status']} (weight {wn:.2f}, snap share {POOL.loc[i, 'snap_share']:.2f})"
              + ("  [QB: if he is out, also set the replacement in qb_overrides.csv]" if g == "QB" and wn >= 0.85 else ""))

teams = sorted(set(tg["team"]))
ADV = {}                                                     # each team's advantage per feature
for name, unit, col, sign in SPEC:
    if col is None:
        ADV[name] = qb_post - qb_post.mean()
    elif name == "qb_swap":                                  # current starter vs the QB the team's stats were built with
        _cur = current_rating("qb_pre")
        ADV[name] = qb_post.reindex(_cur.index) - (_cur + tg["qb_pre"].mean())
    else:
        ADV[name] = sign * current_rating(col, unit=unit if RET_PRIOR else None)
def burden(t, groups):
    return float(sum(bw.loc[t, g] for g in groups if t in bw.index and g in bw.columns))
for name, unit, groups in INJ:
    ADV[name] = pd.Series({t: -burden(t, groups) for t in teams})
for name, unit, pos, col in CONT_DEF:                        # each team's lineup continuity going into this week
    if name in CONT_NAMES:
        ADV[name] = pd.Series(cont_lookup(CONT_TAB[name], [season_now] * len(teams), [wk] * len(teams), teams), index=teams)
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
coef.rename("coef").rename_axis("feature").reset_index().to_csv(f"{BOARD_DIR}/model_weights_latest.csv", index=False)
if len(POOL):
    POOL.drop(columns=["key"]).to_csv(f"{BOARD_DIR}/injury_player_pool_latest.csv", index=False)
log(f"done. Saved the dated board, 'latest' copies and team ratings to {BOARD_DIR}")
print("\nWeekly routine: run after the Wednesday and Friday injury reports. Each dated board file is a "
      "pregame record (it carries its own build time and each game's kickoff time).")
