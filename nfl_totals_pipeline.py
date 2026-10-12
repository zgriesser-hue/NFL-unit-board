# NFL game-totals (over/under) pipeline: data -> team ratings -> totals model -> weekly totals board.
# Companion to nfl_spread_pipeline.py. Same repo layout and conventions:
#   * runs on GitHub Actions / locally; reads optional data/game_lines.csv; writes dated boards to boards/
#   * optional ODDS_API_KEY secret captures the current market total (1 Odds API credit per run)
# Output: boards/totals_board_latest.csv (+ a dated copy) and boards/totals_backtest_latest.csv.
#
# Model: ridge regression of final combined score on pregame team ratings (EPA/play by pass and rush, offense and
# defense), pace, weather and context. Ratings blend last season with this season's games so far (same idea as the
# spread model). The board splits the fair total into units, like the unit board:
#   league average + QB + Offense + Defense + Pace + Weather + Context = fair total.
# QB: each team's starter is rated by his shrunk EPA per dropback. "qb_swap" is the starter versus the QBs the team's
# stats were built with. For the upcoming slate the starter is last game's QB; if he left that game early the model blends
# him with his replacement (TOT_QB_P_OUT, default 50/50); qb_overrides.csv (same file as the spread model) wins.
# The honest question is whether it beats the CLOSING total, so the run prints a walk-forward check against it.

import os
import re
import sys
import time
import subprocess

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# ================================================================== settings
ROOT = os.environ.get("NFL_ROOT", os.getcwd())
DATA_DIR = f"{ROOT}/data"
BOARD_DIR = f"{ROOT}/boards"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
FIRST_SEASON, LAST_SEASON, PBP_FROM = 2020, 2026, 2019
TEST_SEASONS = [2023, 2024, 2025]
K_PRIOR = float(os.environ.get("TOT_K_PRIOR", "8"))          # a team's rating starts as K games of prior
PRIOR_W = float(os.environ.get("TOT_PRIOR_W", "0.6"))        # prior = this share last season's team mean, rest league mean
DECAY = float(os.environ.get("TOT_DECAY", "1.0"))            # in-season game weight: 1.0 = equal, 0.9 = older games count 10% less
ALPHA = float(os.environ.get("TOT_ALPHA", "100"))            # ridge strength on standardized features
WEEK_MODE = os.environ.get("TOT_WEEK_MODE", "")              # "next" = Monday-morning run: build NEXT week's slate (the first week with no games played yet)
USE_FORECAST = os.environ.get("TOT_FORECAST", "1") != "0"      # fill missing game weather from a free forecast (Open-Meteo); set TOT_FORECAST=0 to turn off
BET_EDGES = [1.0, 2.0, 3.0]                                  # |model - closing total| thresholds for the over/under record

K_QB = 150.0                                                 # a QB's rating starts as this many dropbacks of a first-year QB
QB_P_OUT = float(os.environ.get("TOT_QB_P_OUT", "0.5"))      # chance a QB who left last week's game early is out this week
RATE_COLS = ["off_pass_epa", "off_rush_epa", "def_pass_epa", "def_rush_epa", "plays", "pass_rate", "qb"]
GROUPS = {"QB": ["qb_level", "qb_swap"], "Offense": ["off_pass", "off_rush"], "Defense": ["def_pass", "def_rush"],
          "Pace": ["pace", "pass_rate"], "Weather": ["dome", "cold", "windy"], "Context": ["neutral", "div"]}
FEATS = [f for fs in GROUPS.values() for f in fs]

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

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}")


# ================================================================== stage 1: games
def build_games(sched, lines=None):
    """One row per regular-season/playoff game with the final total and the market totals we have."""
    g = sched[sched["season"] >= PBP_FROM].copy()          # 2019 only feeds the first priors; training starts at FIRST_SEASON
    g["kickoff_utc"] = (pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"), format="%Y-%m-%d %H:%M")
                          .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward")
                          .dt.tz_convert("UTC"))
    g["is_playoff"] = (g["game_type"] != "REG").astype(int)
    g["neutral"] = (g["location"] == "Neutral").astype(float)
    g["div"] = g["div_game"].fillna(0).astype(float)
    g["total_pts"] = g["home_score"] + g["away_score"]
    g = g.rename(columns={"total_line": "total_sched", "temp": "temp_actual", "wind": "wind_actual"})
    g["mkt_total_close"], g["mkt_total_open"] = g["total_sched"], np.nan      # nflverse closing total unless better lines exist
    if lines is not None and {"game_id", "total_close"}.issubset(lines.columns):
        cols = [c for c in ["game_id", "total_open_wk", "total_close"] if c in lines.columns]
        g = g.merge(lines[cols], on="game_id", how="left")
        g["mkt_total_close"] = g["total_close"].fillna(g["mkt_total_close"])
        if "total_open_wk" in g:
            g["mkt_total_open"] = g["total_open_wk"]
        g = g.drop(columns=[c for c in ["total_close", "total_open_wk"] if c in g.columns])
    return g.sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)


# ================================================================== stage 2: team-games from play by play
def build_team_games(pbp, games):
    """Per team per game: EPA/play on offense and defense (pass and rush), offensive plays, pass rate."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna() & pbp["posteam"].notna()].copy()
    p["is_pass"] = (p["play_type"] == "pass").astype(float)
    ps, rs = p[p["play_type"] == "pass"], p[p["play_type"] == "run"]
    off = (p.groupby(["game_id", "posteam"]).agg(plays=("epa", "size"), pass_rate=("is_pass", "mean"))
             .reset_index().rename(columns={"posteam": "team"}))
    parts = [
        ps.groupby(["game_id", "posteam"]).agg(off_pass_epa=("epa", "mean")).reset_index().rename(columns={"posteam": "team"}),
        rs.groupby(["game_id", "posteam"]).agg(off_rush_epa=("epa", "mean")).reset_index().rename(columns={"posteam": "team"}),
        ps.groupby(["game_id", "defteam"]).agg(def_pass_epa=("epa", "mean")).reset_index().rename(columns={"defteam": "team"}),
        rs.groupby(["game_id", "defteam"]).agg(def_rush_epa=("epa", "mean")).reset_index().rename(columns={"defteam": "team"}),
    ]
    for part in parts:
        off = off.merge(part, on=["game_id", "team"], how="outer")
    info = games[["game_id", "season", "week", "kickoff_utc", "home_team", "away_team", "is_playoff"]]
    tg = off.merge(info, on="game_id", how="inner")
    tg["is_home"] = (tg["team"] == tg["home_team"]).astype(int)
    tg = tg[tg["is_playoff"] == 0].sort_values(["team", "kickoff_utc"]).reset_index(drop=True)
    return tg


# ================================================================== stage 2b: quarterbacks
def _qid(x):
    """Name -> 'jdaniels' (first initial + last name) so 'Jayden Daniels', 'J.Daniels' and 'Daniels' can match."""
    x = str(x).strip()
    if "." in x and " " not in x:
        parts = x.split(".", 1)
    else:
        parts = [p for p in re.sub(r"[^A-Za-z ]", "", x).split() if p.lower() not in ("jr", "sr", "ii", "iii", "iv")]
    parts = [re.sub(r"[^a-z]", "", p.lower()) for p in parts if p]
    if not parts:
        return ""
    return (parts[0][:1] + parts[-1]) if len(parts) > 1 else parts[0]


def build_qb_tables(pbp, games):
    """qg: one row per QB per team-game (dropbacks, EPA, his rating going INTO the game). qcur: each QB's rating now.
    starters: the QB with the most dropbacks in each team-game, with his pregame rating (column qb)."""
    q = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna() & pbp["qb_epa"].notna() & pbp["posteam"].notna()]
    qg = (q.groupby(["game_id", "posteam", "passer_player_id"])
            .agg(name=("passer_player_name", "first"), db=("qb_epa", "size"), epa_sum=("qb_epa", "sum"),
                 first_play=("play_id", "min"), last_play=("play_id", "max"))
            .reset_index().rename(columns={"posteam": "team"}))
    qg = qg.merge(games[["game_id", "kickoff_utc"]], on="game_id", how="inner").sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)
    first = qg.groupby("passer_player_id").head(1)
    prior = float(first["epa_sum"].sum() / first["db"].sum())            # what a QB with no record usually produces
    g = qg.groupby("passer_player_id")
    qg["pre"] = ((g["epa_sum"].cumsum() - qg["epa_sum"]) + K_QB * prior) / ((g["db"].cumsum() - qg["db"]) + K_QB)
    qg["post"] = (g["epa_sum"].cumsum() + K_QB * prior) / (g["db"].cumsum() + K_QB)     # his rating AFTER this game
    tot = qg.groupby("passer_player_id").agg(name=("name", "last"), team=("team", "last"), db=("db", "sum"), epa=("epa_sum", "sum"))
    tot["rating"] = (tot["epa"] + K_QB * prior) / (tot["db"] + K_QB)
    tot["key"] = tot["name"].map(_qid)
    qcur = tot.reset_index()
    starters = (qg.sort_values("db", ascending=False).drop_duplicates(["game_id", "team"])
                  [["game_id", "team", "pre", "name"]].rename(columns={"pre": "qb", "name": "qb_name"}))
    return qg, qcur, starters


def _starter_from_game(lg, rating, next_kick):
    """(rating, name, note) assumed for a team's NEXT game from the QBs of its last game. rating maps passer id -> rating."""
    main = lg.sort_values("db", ascending=False).iloc[0]
    early, late = lg.sort_values("first_play").iloc[0], lg.sort_values("last_play").iloc[-1]
    recent = (next_kick - lg["kickoff_utc"].iloc[0]) <= pd.Timedelta(days=10)           # a bye week lets a knock heal
    if recent and len(lg) > 1 and early["passer_player_id"] != late["passer_player_id"] and early["db"] / lg["db"].sum() < 0.8:
        r_ = (1 - QB_P_OUT) * rating[early["passer_player_id"]] + QB_P_OUT * rating[late["passer_player_id"]]
        return (float(r_), f"{early['name']}/{late['name']}",
                f"CHECK: {early['name']} left last game early ({1 - QB_P_OUT:.0%} {early['name']}, {QB_P_OUT:.0%} {late['name']})")
    return float(rating[main["passer_player_id"]]), main["name"], "last starter"


def monday_starters(qg):
    """For every historical team-game: the QB rating you would have assumed on Monday, before any starter news
    (last game's QB, or the early-exit blend). Used to test what Monday's board could really have known."""
    rows = []
    for team, tgrp in qg.groupby("team"):
        prev = None
        for gid, lg in tgrp.groupby("game_id", sort=False):                # time order: qg is sorted by kickoff
            if prev is not None:
                r_, _, _ = _starter_from_game(prev, dict(zip(prev["passer_player_id"], prev["post"])), lg["kickoff_utc"].iloc[0])
                rows.append((gid, team, r_))
            prev = lg
    return pd.DataFrame(rows, columns=["game_id", "team", "qb_mon"])


def pick_starters(qg, qcur, up, overrides=None):
    """Expected starting QB for each team in the upcoming slate -> {team: (rating, name, note)}."""
    now = qcur.set_index("passer_player_id")["rating"]
    repl = float(qcur.loc[qcur["db"] >= 300, "rating"].quantile(0.10))        # replacement level: a weak current starter
    kick = {}
    for _, r in up.iterrows():
        kick[r["home_team"]], kick[r["away_team"]] = r["kickoff_utc"], r["kickoff_utc"]
    out = {}
    for team, ko in kick.items():
        grp = qg[qg["team"] == team]
        if grp.empty:
            out[team] = (repl, "unknown", "no QB history: replacement level")
            continue
        lg = grp[grp["game_id"] == grp.iloc[-1]["game_id"]]                    # qg is in time order: last game's QBs
        out[team] = _starter_from_game(lg, now, ko)
    if overrides is not None and len(overrides):
        for _, o in overrides.iterrows():
            t = str(o["team"]).strip().upper()
            if t not in out:
                continue
            k = _qid(o["qb"])
            cand = qcur[qcur["key"] == k] if k else qcur.iloc[0:0]
            if cand.empty and k:
                cand = qcur[qcur["key"].str.endswith(k)]
            if len(cand) and (cand["team"] == t).any():
                cand = cand[cand["team"] == t]
            if cand.empty:
                out[t] = (repl, str(o["qb"]).strip(), "override: no track record, replacement level")
            else:
                best = cand.sort_values("db").iloc[-1]
                out[t] = (float(best["rating"]), str(o["qb"]).strip(), "override")
    return out


# ================================================================== stage 3: pregame ratings
def _prior(team_season_mean, league_mean, team, season, col):
    """Starting value for a team-season: part last season's team mean, part last season's league mean."""
    lm = league_mean.get(season - 1)
    if lm is None:
        return np.nan
    tm = team_season_mean.get((team, season - 1))
    return lm[col] if tm is None else PRIOR_W * tm[col] + (1 - PRIOR_W) * lm[col]


def add_pregame_ratings(tg, teams_now, season_now):
    """Adds <col>_pre to every team-game (rating BEFORE that game, nothing from the game itself), and returns the
    current rating of each team in season_now (after every completed game) for the upcoming slate."""
    tg = tg.sort_values(["team", "kickoff_utc"]).reset_index(drop=True)
    tsm = {k: v for k, v in tg.groupby(["team", "season"])[RATE_COLS].mean().iterrows()}
    lgm = {k: v for k, v in tg.groupby("season")[RATE_COLS].mean().iterrows()}
    for c in RATE_COLS:
        tg[c + "_pre"] = np.nan
    cur = {}
    for (team, season), idx in tg.groupby(["team", "season"]).indices.items():
        for c in RATE_COLS:
            prior = _prior(tsm, lgm, team, season, c)
            vals = tg[c].values
            acc = n = 0.0
            pre = np.full(len(idx), np.nan)
            for j, i in enumerate(idx):
                pre[j] = (acc + K_PRIOR * prior) / (n + K_PRIOR) if np.isfinite(prior) else np.nan
                if np.isfinite(vals[i]):
                    acc, n = DECAY * acc + vals[i], DECAY * n + 1.0
                else:
                    acc, n = DECAY * acc, DECAY * n
            tg.loc[tg.index[idx], c + "_pre"] = pre
            if season == season_now:
                cur[(team, c)] = (acc + K_PRIOR * prior) / (n + K_PRIOR) if np.isfinite(prior) else np.nan
    for team in teams_now:                              # a team with no games yet this season starts at its prior
        for c in RATE_COLS:
            if (team, c) not in cur:
                cur[(team, c)] = _prior(tsm, lgm, team, season_now, c)
    return tg, cur


# ================================================================== stage 4: game features
def make_features(df, qcol="qb_now"):
    """df has h_<col> and a_<col> rating columns plus roof/temp/wind/neutral/div. Returns the model features."""
    X = pd.DataFrame(index=df.index)
    X["off_pass"] = df["h_off_pass_epa"] + df["a_off_pass_epa"]
    X["off_rush"] = df["h_off_rush_epa"] + df["a_off_rush_epa"]
    X["def_pass"] = df["h_def_pass_epa"] + df["a_def_pass_epa"]       # EPA allowed: higher = worse defense = more points
    X["def_rush"] = df["h_def_rush_epa"] + df["a_def_rush_epa"]
    X["qb_level"] = df[f"h_{qcol}"] + df[f"a_{qcol}"]                   # the two starters' ratings
    X["qb_swap"] = (df[f"h_{qcol}"] - df["h_qb"]) + (df[f"a_{qcol}"] - df["a_qb"])     # starter vs the QBs the team's stats were built with
    X["pace"] = df["h_plays"] + df["a_plays"]
    X["pass_rate"] = (df["h_pass_rate"] + df["a_pass_rate"]) / 2
    dome = df["roof"].isin(["dome", "closed"])
    X["dome"] = dome.astype(float)
    temp = df["temp_actual"].where(~dome, 70.0).fillna(65.0)          # unknown outdoor weather = mild
    wind = df["wind_actual"].where(~dome, 0.0).fillna(0.0)
    X["cold"] = (45.0 - temp).clip(lower=0)
    X["windy"] = (wind - 10.0).clip(lower=0)
    X["neutral"] = df["neutral"].astype(float)
    X["div"] = df["div"].astype(float)
    return X[FEATS]


def historical_table(games, tg):
    h = tg[tg["is_home"] == 1][["game_id"] + [c + "_pre" for c in RATE_COLS] + ["qb", "qb_mon"]]
    a = tg[tg["is_home"] == 0][["game_id"] + [c + "_pre" for c in RATE_COLS] + ["qb", "qb_mon"]]
    h.columns = ["game_id"] + ["h_" + c for c in RATE_COLS] + ["h_qb_now", "h_qb_mon"]   # h_qb = QBs so far; _now = who actually started; _mon = what Monday would assume
    a.columns = ["game_id"] + ["a_" + c for c in RATE_COLS] + ["a_qb_now", "a_qb_mon"]
    d = games[(games["is_playoff"] == 0) & (games["season"] >= FIRST_SEASON)].merge(h, on="game_id").merge(a, on="game_id")
    d = d[d["total_pts"].notna()].copy()
    X = make_features(d)
    Xm = make_features(d, qcol="qb_mon")                  # same features, but with the starter Monday would have assumed
    ok = X.notna().all(axis=1)
    return d[ok].reset_index(drop=True), X[ok].reset_index(drop=True), Xm[ok].reset_index(drop=True)


# ================================================================== stage 5: model
RESID_ALPHAS = [300, 1000, 3000]          # shrinkage strengths tried for the market-residual model


class TotalsModel:
    def __init__(self, alpha=ALPHA):
        self.alpha = alpha

    def fit(self, X, y):
        self.mu, self.sd = X.mean(), X.std().replace(0, 1.0)
        self.ridge = Ridge(alpha=self.alpha).fit(((X - self.mu) / self.sd).values, y.values)
        self.coef = pd.Series(self.ridge.coef_, index=X.columns)
        self.base = float(self.ridge.intercept_)
        return self

    def parts(self, X):
        """League average plus each unit's contribution in points; the columns sum to the fair total."""
        z = (X - self.mu) / self.sd
        out = pd.DataFrame(index=X.index)
        out["league_avg"] = self.base
        for unit, feats in GROUPS.items():
            feats = [f for f in feats if f in self.coef.index]
            out[unit] = z[feats].values @ self.coef[feats].values if feats else 0.0
        return out

    def predict(self, X):
        return self.parts(X).sum(axis=1)


def over_under_record(edge, actual, line, thresholds=BET_EDGES):
    rows = []
    for t in thresholds:
        m = (edge.abs() >= t) & (actual != line)                       # pushes excluded
        n = int(m.sum())
        over = edge[m] > 0
        win = (over & (actual[m] > line[m])) | (~over & (actual[m] < line[m]))
        rows.append({"edge_at_least": t, "games": n, "win_pct": float(win.mean()) if n else np.nan})
    return pd.DataFrame(rows)


def backtest(d, X, Xmon):
    """Walk-forward: for each test season train only on earlier seasons, predict that season."""
    out = []
    for s in TEST_SEASONS:
        tr, te = d["season"] < s, d["season"] == s
        if tr.sum() < 200 or te.sum() == 0:
            continue
        m = TotalsModel().fit(X[tr], d.loc[tr, "total_pts"])
        t = d[te][["game_id", "season", "week", "total_pts", "mkt_total_close", "mkt_total_open"]].copy()
        t["model"] = m.predict(X[te]).values
        Xn = X.copy()                                  # same model with the observed wind and cold removed: those are not
        Xn[["cold", "windy"]] = 0.0                    # known when the opening total is posted (roof/dome is)
        m_nw = TotalsModel().fit(Xn[tr], d.loc[tr, "total_pts"])
        t["model_nw"] = m_nw.predict(Xn[te]).values
        Xmn = Xmon.copy()                              # AS OF MONDAY: the starter you would have assumed (not the one who played)
        Xmn[["cold", "windy"]] = 0.0                   # and no observed wind/cold
        t["model_mon"] = m_nw.predict(Xmn[te]).values
        # market-residual model: learn only what the opening total misses (actual - open), shrunk hard toward zero.
        # Trained on games with an opening line, no observed wind/cold, scored with the as-of-Monday starters.
        hasopen = d["mkt_total_open"].notna()
        for a in RESID_ALPHAS:
            tro = tr & hasopen
            col = f"resid_a{a}"
            t[col] = np.nan
            if tro.sum() >= 200:
                mr = TotalsModel(alpha=a).fit(Xn[tro], d.loc[tro, "total_pts"] - d.loc[tro, "mkt_total_open"])
                t[col] = (d.loc[te, "mkt_total_open"] + mr.predict(Xmn[te])).values
        Xq = X.drop(columns=GROUPS["QB"])                # the same model with no QB information, to see what the QB features add
        t["model_noqb"] = TotalsModel().fit(Xq[tr], d.loc[tr, "total_pts"]).predict(Xq[te]).values
        t["naive"] = float(d.loc[tr, "total_pts"].mean())
        out.append(t)
    if not out:
        return None, None
    r = pd.concat(out)
    r = r[r["mkt_total_close"].notna()].copy()
    r["blend"] = 0.5 * r["model"] + 0.5 * r["mkt_total_close"]
    r["edge"] = r["model"] - r["mkt_total_close"]
    mae = lambda c: float((r[c] - r["total_pts"]).abs().mean())
    summ = pd.DataFrame([{"seasons": str(TEST_SEASONS), "games": len(r), "mae_model": mae("model"),
                          "mae_closing_total": mae("mkt_total_close"), "mae_blend_50_50": mae("blend"),
                          "mae_league_avg": mae("naive"), "mae_model_no_qb": mae("model_noqb"),
                          "corr_edge_vs_miss": float(np.corrcoef(r["edge"], r["total_pts"] - r["mkt_total_close"])[0, 1])}])
    return r, summ


def open_line_check(r, col="model", thresholds=BET_EDGES):
    """Does the model's disagreement with the OPENING total predict where the line moves by close, and would betting
    the open have won? Moving toward the model is closing-line value: the usual sign of a real edge."""
    o = r[r["mkt_total_open"].notna() & r[col].notna()].copy()
    if len(o) < 100:
        return None, None
    o["edge_open"] = o[col] - o["mkt_total_open"]
    o["move"] = o["mkt_total_close"] - o["mkt_total_open"]
    corr = float(np.corrcoef(o["edge_open"], o["move"])[0, 1])
    rec = over_under_record(o["edge_open"], o["total_pts"], o["mkt_total_open"], thresholds)
    extra = []
    for t in thresholds:
        m = (o["edge_open"].abs() >= t) & (o["move"] != 0)
        toward = np.sign(o.loc[m, "move"]) == np.sign(o.loc[m, "edge_open"])
        gained = o.loc[m, "move"] * np.sign(o.loc[m, "edge_open"])        # points the line moved toward the model's side
        extra.append({"edge_at_least": t, "moved_games": int(m.sum()),
                      "moved_toward_model_pct": float(toward.mean()) if m.sum() else np.nan,
                      "avg_points_toward_model": float(gained.mean()) if m.sum() else np.nan})
    rec = rec.merge(pd.DataFrame(extra), on="edge_at_least")
    return {"open_games": len(o), "corr_edge_vs_move": corr}, rec


# ================================================================== stage 5b: forecast weather
STADIUMS = {   # home venue (latitude, longitude); a few km is plenty for a forecast
    "ARI": (33.5276, -112.2626), "ATL": (33.7554, -84.4009), "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167), "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201), "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639), "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LV": (36.0909, -115.1833), "LAC": (33.9535, -118.3392), "LA": (33.9535, -118.3392), "MIA": (25.9580, -80.2389),
    "MIN": (44.9738, -93.2577), "NE": (42.0909, -71.2643), "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "PIT": (40.4468, -80.0158), "SF": (37.4030, -121.9697),
    "SEA": (47.5952, -122.3316), "TB": (27.9759, -82.5033), "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645)}


def fetch_forecast_weather(up):
    """Forecast temperature (F) and wind (mph) at kickoff + 90 minutes for outdoor games that have no weather yet.
    Free Open-Meteo forecast, no key. Returns {game_id: (temp, wind)}. Neutral-site games are skipped (venue unknown)."""
    import requests
    out = {}
    need_all = up[~up["roof"].isin(["dome", "closed"]) & up["temp_actual"].isna() & (up["neutral"] == 0)]
    for team, grp in need_all.groupby("home_team"):
        if team not in STADIUMS:
            continue
        lat, lon = STADIUMS[team]
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={"latitude": lat, "longitude": lon, "hourly": "temperature_2m,wind_speed_10m",
                                 "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                                 "timezone": "UTC", "forecast_days": 16}, timeout=30)
        r.raise_for_status()
        h = r.json()["hourly"]
        times = pd.to_datetime(pd.Series(h["time"]), utc=True)
        for _, g in grp.iterrows():
            gt = g["kickoff_utc"] + pd.Timedelta(minutes=90)
            gap = (times - gt).abs()
            i = int(gap.idxmin())
            t, w = h["temperature_2m"][i], h["wind_speed_10m"][i]
            if gap[i] <= pd.Timedelta(hours=2) and t is not None and w is not None:
                out[g["game_id"]] = (float(t), float(w))
    return out


# ================================================================== stage 6: market totals
def fetch_market_totals(key):
    """Consensus (median across books) current game total. Costs 1 Odds API credit."""
    import requests
    r = requests.get("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
                     params={"apiKey": key, "regions": "us", "markets": "totals", "oddsFormat": "american"}, timeout=60)
    r.raise_for_status()
    rows = []
    for ev in r.json():
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        pts = [oc["point"] for bk in ev.get("bookmakers", []) for mk in bk.get("markets", [])
               if mk["key"] == "totals" for oc in mk["outcomes"]
               if oc["name"] == "Over" and -150 <= oc.get("price", -110) <= 130]
        dk = [oc["point"] for bk in ev.get("bookmakers", []) if bk.get("key") == "draftkings"
              for mk in bk.get("markets", []) if mk["key"] == "totals" for oc in mk["outcomes"] if oc["name"] == "Over"]
        if home and away and pts:
            # every book's Over/Under (point, price, book) for line shopping
            ov, un = [], []
            for bk in ev.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    if mk["key"] != "totals":
                        continue
                    for oc in mk["outcomes"]:
                        if oc.get("point") is None or not -150 <= oc.get("price", -110) <= 130:
                            continue
                        rec = (float(oc["point"]), int(oc["price"]), bk.get("title") or bk.get("key"))
                        if oc["name"] == "Over":
                            ov.append(rec)
                        elif oc["name"] == "Under":
                            un.append(rec)
            row = {"home_team": home, "away_team": away, "market_total_api": float(np.median(pts)),
                   "market_books": len(pts), "dk_total": float(dk[0]) if dk else np.nan}
            if ov:   # best Over = lowest number, then best price
                b = min(ov, key=lambda r: (r[0], -r[1]))
                row.update(best_over_line=b[0], best_over_price=b[1], best_over_book=b[2])
            if un:   # best Under = highest number, then best price
                b = max(un, key=lambda r: (r[0], r[1]))
                row.update(best_under_line=b[0], best_under_price=b[1], best_under_book=b[2])
            rows.append(row)
    return pd.DataFrame(rows)


# ================================================================== main
def main():
    try:
        import nflreadpy as nfl
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nflreadpy"], check=False)
        import nflreadpy as nfl
    os.makedirs(BOARD_DIR, exist_ok=True)

    log("loading schedules ...")
    sched = nfl.load_schedules(seasons=list(range(PBP_FROM, LAST_SEASON + 1))).to_pandas()
    gl = f"{DATA_DIR}/game_lines.csv"
    lines = pd.read_csv(gl) if os.path.exists(gl) else None
    print(f"    game_lines.csv found: {lines is not None} (looked in {DATA_DIR})"
          + ("" if lines is not None else "; using nflverse closing totals for the check"))
    games = build_games(sched, lines)

    log("loading play-by-play (a minute or two) ...")
    pbp = (nfl.load_pbp(seasons=list(range(PBP_FROM, LAST_SEASON + 1)))
              .select(["game_id", "posteam", "defteam", "play_type", "epa", "play_id", "qb_dropback", "qb_epa",
                        "passer_player_id", "passer_player_name"]).to_pandas())
    tg = build_team_games(pbp, games)
    qg, qcur, qstart = build_qb_tables(pbp, games)
    tg = tg.merge(qstart[["game_id", "team", "qb"]], on=["game_id", "team"], how="left")
    tg = tg.merge(monday_starters(qg), on=["game_id", "team"], how="left")

    played = games[games["total_pts"].notna() & (games["is_playoff"] == 0)]
    season_now = int(played["season"].max())
    todo = games[games["total_pts"].isna() & (games["is_playoff"] == 0) & (games["season"] >= season_now)]
    wk = int(todo["week"].min()) if len(todo) else None
    if WEEK_MODE == "next" and len(todo):                 # Monday morning: this week's Monday-night game is still unplayed,
        _cur = games[(games["season"] == season_now) & (games["is_playoff"] == 0)]    # so skip ahead to a week with nothing played
        _done = _cur.groupby("week")["total_pts"].apply(lambda x: int(x.notna().sum()))
        _fresh = sorted(w for w in todo["week"].unique() if _done.get(w, 0) == 0)
        if _fresh:
            wk = int(_fresh[0])
    _no_tape = games[(games["season"] == season_now) & (games["is_playoff"] == 0) & games["total_pts"].notna()
                     & ~games["game_id"].isin(set(tg["game_id"]))]
    if len(_no_tape):                                     # scores are in the schedule but play-by-play has not caught up yet
        print(f"    WARNING: {len(_no_tape)} finished game(s) not in play-by-play yet, so ratings leave them out: "
              + ", ".join((_no_tape["away_team"] + "@" + _no_tape["home_team"]).tolist()[:8]))
    teams_now = sorted(set(games.loc[games["season"] == season_now, "home_team"]) |
                       set(games.loc[games["season"] == season_now, "away_team"]))
    tg, cur = add_pregame_ratings(tg, teams_now, season_now)
    d, X, Xmon = historical_table(games, tg)
    log(f"{len(d)} completed games with full pregame ratings ({d['season'].min()}-{d['season'].max()})")

    # ---- honest out-of-sample check against the closing total
    r, summ = backtest(d, X, Xmon)
    if summ is not None:
        s = summ.iloc[0]
        print(f"\nOUT-OF-SAMPLE {TEST_SEASONS}, {int(s['games'])} games, mean miss in points: model {s['mae_model']:.3f} | "
              f"closing total {s['mae_closing_total']:.3f} | 50/50 blend {s['mae_blend_50_50']:.3f} | league average {s['mae_league_avg']:.3f}")
        print(f"  without the QB features the model misses by {s['mae_model_no_qb']:.3f} (with them {s['mae_model']:.3f}; lower is better)")
        print(f"  correlation of (model - closing total) with (actual - closing total): {s['corr_edge_vs_miss']:+.3f} "
              f"(0 = our disagreement with the market carries no information)")
        rec = over_under_record(r["edge"], r["total_pts"], r["mkt_total_close"])
        print("  over/under record when the model disagrees with the closing total (pushes excluded; need > 52.4% to beat -110):")
        print(rec.round(3).to_string(index=False))
        summ.to_csv(f"{BOARD_DIR}/totals_backtest_latest.csv", index=False)
        rec.to_csv(f"{BOARD_DIR}/totals_backtest_record_latest.csv", index=False)
        oinfo, orec = open_line_check(r)
        if oinfo is not None:
            print(f"\nOPENING-LINE CHECK, {oinfo['open_games']} games, DIAGNOSTIC ONLY: it knows the QB who actually played and the observed "
                  f"weather, which Monday's board cannot know, so it flatters the model:")
            print(f"  correlation of (model - opening total) with (closing - opening total): {oinfo['corr_edge_vs_move']:+.3f} "
                  f"(positive = the market tends to move toward the model)")
            print(orec.round(3).to_string(index=False))
            orec.to_csv(f"{BOARD_DIR}/totals_open_line_record_latest.csv", index=False)
            ninfo, nrec = open_line_check(r, col="model_mon")
            if ninfo is not None:
                print("\nAS OF MONDAY (the honest check): last game's QB with the early-exit blend, no observed wind or cold. "
                      "This is what the Monday board could really have known:")
                print(f"  correlation of (model - opening total) with (closing - opening total): {ninfo['corr_edge_vs_move']:+.3f}")
                print("  betting the OPENING total on the model's side (need > 52.4% at -110), and how the line then moved:")
                print(nrec.round(3).to_string(index=False))
                nrec.to_csv(f"{BOARD_DIR}/totals_open_line_record_monday_latest.csv", index=False)
            # market-residual model (learns only what the opening total misses), same honest as-of-Monday inputs
            allrec = []
            for a in RESID_ALPHAS:
                col = f"resid_a{a}"
                if col not in r:
                    continue
                rinfo, rrec = open_line_check(r, col=col, thresholds=[0.0, 0.5, 1.0, 1.5, 2.0])
                if rinfo is None:
                    continue
                ro = r[r["mkt_total_open"].notna() & r[col].notna()]
                print(f"\nMARKET-RESIDUAL MODEL (shrinkage alpha={a}), {rinfo['open_games']} games: mean miss {float((ro[col] - ro['total_pts']).abs().mean()):.3f} "
                      f"vs opening total {float((ro['mkt_total_open'] - ro['total_pts']).abs().mean()):.3f}; "
                      f"corr with the line's move {rinfo['corr_edge_vs_move']:+.3f}")
                print(rrec.round(3).to_string(index=False))
                allrec.append(rrec.assign(alpha=a))
            if allrec:
                pd.concat(allrec).to_csv(f"{BOARD_DIR}/totals_resid_record_latest.csv", index=False)
        else:
            print("\n(no opening totals in game_lines.csv for the test seasons: skipped the opening-line check)")
    else:
        print("\n(not enough history for the out-of-sample check)")

    if wk is None:
        print("No upcoming regular-season games found; nothing to put on the board.")
        return
    model = TotalsModel().fit(X, d["total_pts"])
    print("\nMODEL WEIGHTS (points of total per 1 standard deviation of the feature):")
    print(model.coef.round(2).to_string())

    up = todo[todo["week"] == wk].copy()
    for side, col in [("h", "home_team"), ("a", "away_team")]:
        for c in RATE_COLS:
            up[f"{side}_{c}"] = up[col].map(lambda t, c=c: cur.get((t, c), np.nan))
    qov_path = f"{ROOT}/qb_overrides.csv"                  # columns: team, qb, optional week (same file the spread model reads)
    qov = pd.read_csv(qov_path) if os.path.exists(qov_path) else None
    if qov is not None and "week" in qov:
        qov = qov[qov["week"].isna() | (qov["week"] == wk)]
    starters = pick_starters(qg, qcur, up, qov)
    for side, col in [("h", "home_team"), ("a", "away_team")]:
        up[f"{side}_qb_now"] = up[col].map(lambda t: starters[t][0])
        up[f"{side}_qb_name"] = up[col].map(lambda t: starters[t][1])
        up[f"{side}_qb_note"] = up[col].map(lambda t: starters[t][2])
    for _, r in up.iterrows():
        for side in ("h", "a"):
            if r[f"{side}_qb_note"].startswith(("CHECK", "override")):
                print(f"  QB {r['home_team'] if side == 'h' else r['away_team']}: {r[f'{side}_qb_name']} - {r[f'{side}_qb_note']}")
    up["weather_src"] = np.where(up["roof"].isin(["dome", "closed"]), "dome",
                                 np.where(up["temp_actual"].notna(), "observed", "none"))
    if USE_FORECAST:
        try:
            fc = fetch_forecast_weather(up)
            for gid, (tmp_f, wind_mph) in fc.items():
                m = up["game_id"] == gid
                up.loc[m, "temp_actual"], up.loc[m, "wind_actual"], up.loc[m, "weather_src"] = tmp_f, wind_mph, "forecast"
            log(f"forecast weather filled for {len(fc)} outdoor game(s)")
        except Exception as e:
            print("Could not fetch forecast weather (games stay at mild defaults):", str(e)[:100])
    Xu = make_features(up)
    weather_known = up["temp_actual"].notna() | up["roof"].isin(["dome", "closed"])
    ok = Xu.notna().all(axis=1)
    if (~ok).any():
        print(f"    skipped (missing ratings): {', '.join((up.loc[~ok, 'away_team'] + '@' + up.loc[~ok, 'home_team']).tolist())}")
    up, Xu, weather_known = up[ok], Xu[ok], weather_known[ok]
    parts = model.parts(Xu)

    board = up[["game_id", "home_team", "away_team", "kickoff_utc"]].copy()
    board["built_at_utc"] = pd.Timestamp.now(tz="UTC")
    for c in ["league_avg"] + list(GROUPS):
        board[c] = parts[c].values
    board["fair_total"] = parts.sum(axis=1).values
    for side, nm in [("h", "home"), ("a", "away")]:
        board[f"{nm}_QB"], board[f"{nm}_QB_note"] = up[f"{side}_qb_name"].values, up[f"{side}_qb_note"].values
    board["weather_known"] = weather_known.values
    board["weather_src"] = up["weather_src"].values
    board["temp_f"] = np.where(up["roof"].isin(["dome", "closed"]), np.nan, up["temp_actual"]).astype(float)
    board["wind_mph"] = np.where(up["roof"].isin(["dome", "closed"]), np.nan, up["wind_actual"]).astype(float)
    board["market_total"] = up["mkt_total_close"].values
    board["market_src"] = np.where(board["market_total"].notna(), "nflverse schedule line", "")
    if ODDS_API_KEY:
        try:
            mk = fetch_market_totals(ODDS_API_KEY)
            board = board.merge(mk, on=["home_team", "away_team"], how="left")
            got = board["market_total_api"].notna()
            board.loc[got, "market_total"] = board.loc[got, "market_total_api"]
            board.loc[got, "market_src"] = "odds api consensus"
            board["market_captured_utc"] = pd.Timestamp.now(tz="UTC")
            board = board.drop(columns=["market_total_api"])
            log(f"market totals captured for {int(got.sum())} of {len(board)} games")
        except Exception as e:
            print("Could not fetch market totals:", str(e)[:100])
    if "dk_total" not in board:
        board["dk_total"] = np.nan
    board["edge_vs_market"] = board["fair_total"] - board["market_total"]            # + = model likes the over
    board["edge_vs_dk"] = board["fair_total"] - board["dk_total"]
    board["lean"] = np.where(board["edge_vs_market"].isna() | (board["edge_vs_market"].abs() < 0.05), "",
                             np.where(board["edge_vs_market"] > 0, "Over", "Under"))
    # line shopping: the best number/price at any book for the side the model leans
    for c in ["best_over_line", "best_over_price", "best_under_line", "best_under_price"]:
        if c not in board:
            board[c] = np.nan
    for c in ["best_over_book", "best_under_book"]:
        if c not in board:
            board[c] = ""
    over = board["lean"] == "Over"
    under = board["lean"] == "Under"
    board["shop_line"] = np.where(over, board["best_over_line"], np.where(under, board["best_under_line"], np.nan))
    board["shop_price"] = np.where(over, board["best_over_price"], np.where(under, board["best_under_price"], np.nan))
    board["shop_book"] = np.where(over, board["best_over_book"], np.where(under, board["best_under_book"], ""))
    # points gained vs the consensus number (positive = shopping helps)
    board["shop_gain_pts"] = np.where(over, board["market_total"] - board["best_over_line"],
                                      np.where(under, board["best_under_line"] - board["market_total"], np.nan))
    board = board.sort_values("kickoff_utc").reset_index(drop=True)

    pd.set_option("display.width", 250)
    print(f"\nTOTALS BOARD, {season_now} week {wk} (points; columns sum to the fair total):")
    print(board.drop(columns=["game_id", "kickoff_utc", "built_at_utc", "market_captured_utc", "market_src"],
                     errors="ignore").round(1).to_string(index=False))
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%MZ")
    board.to_csv(f"{BOARD_DIR}/totals_board_{season_now}_wk{wk}_{stamp}.csv", index=False)
    board.to_csv(f"{BOARD_DIR}/totals_board_latest.csv", index=False)
    model.coef.rename("coef").rename_axis("feature").reset_index().to_csv(f"{BOARD_DIR}/totals_model_weights_latest.csv", index=False)
    log(f"done. Saved the dated totals board and 'latest' copies to {BOARD_DIR}")


if __name__ == "__main__":
    main()
