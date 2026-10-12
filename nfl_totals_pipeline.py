# NFL game-totals (over/under) pipeline: data -> team ratings -> totals model -> weekly totals board.
# Companion to nfl_spread_pipeline.py. Same repo layout and conventions:
#   * runs on GitHub Actions / locally; reads optional data/game_lines.csv; writes dated boards to boards/
#   * optional ODDS_API_KEY secret captures the current market total (1 Odds API credit per run)
# Output: boards/totals_board_latest.csv (+ a dated copy) and boards/totals_backtest_latest.csv.
#
# Model: ridge regression of final combined score on pregame team ratings (EPA/play by pass and rush, offense and
# defense), pace, weather and context. Ratings blend last season with this season's games so far (same idea as the
# spread model). The board splits the fair total into units, like the unit board:
#   league average + Offense + Defense + Pace + Weather + Context = fair total.
# The honest question is whether it beats the CLOSING total, so the run prints a walk-forward check against it.

import os
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
BET_EDGES = [1.0, 2.0, 3.0]                                  # |model - closing total| thresholds for the over/under record

RATE_COLS = ["off_pass_epa", "off_rush_epa", "def_pass_epa", "def_rush_epa", "plays", "pass_rate"]
GROUPS = {"Offense": ["off_pass", "off_rush"], "Defense": ["def_pass", "def_rush"],
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
def make_features(df):
    """df has h_<col> and a_<col> rating columns plus roof/temp/wind/neutral/div. Returns the model features."""
    X = pd.DataFrame(index=df.index)
    X["off_pass"] = df["h_off_pass_epa"] + df["a_off_pass_epa"]
    X["off_rush"] = df["h_off_rush_epa"] + df["a_off_rush_epa"]
    X["def_pass"] = df["h_def_pass_epa"] + df["a_def_pass_epa"]       # EPA allowed: higher = worse defense = more points
    X["def_rush"] = df["h_def_rush_epa"] + df["a_def_rush_epa"]
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
    h = tg[tg["is_home"] == 1][["game_id"] + [c + "_pre" for c in RATE_COLS]]
    a = tg[tg["is_home"] == 0][["game_id"] + [c + "_pre" for c in RATE_COLS]]
    h.columns = ["game_id"] + ["h_" + c for c in RATE_COLS]
    a.columns = ["game_id"] + ["a_" + c for c in RATE_COLS]
    d = games[(games["is_playoff"] == 0) & (games["season"] >= FIRST_SEASON)].merge(h, on="game_id").merge(a, on="game_id")
    d = d[d["total_pts"].notna()].copy()
    X = make_features(d)
    ok = X.notna().all(axis=1)
    return d[ok].reset_index(drop=True), X[ok].reset_index(drop=True)


# ================================================================== stage 5: model
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
            out[unit] = z[feats].values @ self.coef[feats].values
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


def backtest(d, X):
    """Walk-forward: for each test season train only on earlier seasons, predict that season."""
    out = []
    for s in TEST_SEASONS:
        tr, te = d["season"] < s, d["season"] == s
        if tr.sum() < 200 or te.sum() == 0:
            continue
        m = TotalsModel().fit(X[tr], d.loc[tr, "total_pts"])
        t = d[te][["game_id", "season", "week", "total_pts", "mkt_total_close", "mkt_total_open"]].copy()
        t["model"] = m.predict(X[te]).values
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
                          "mae_league_avg": mae("naive"),
                          "corr_edge_vs_miss": float(np.corrcoef(r["edge"], r["total_pts"] - r["mkt_total_close"])[0, 1])}])
    return r, summ


def open_line_check(r, thresholds=BET_EDGES):
    """Does the model's disagreement with the OPENING total predict where the line moves by close, and would betting
    the open have won? Moving toward the model is closing-line value: the usual sign of a real edge."""
    o = r[r["mkt_total_open"].notna()].copy()
    if len(o) < 100:
        return None, None
    o["edge_open"] = o["model"] - o["mkt_total_open"]
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
        if home and away and pts:
            rows.append({"home_team": home, "away_team": away,
                         "market_total_api": float(np.median(pts)), "market_books": len(pts)})
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
              .select(["game_id", "posteam", "defteam", "play_type", "epa"]).to_pandas())
    tg = build_team_games(pbp, games)

    played = games[games["total_pts"].notna() & (games["is_playoff"] == 0)]
    season_now = int(played["season"].max())
    todo = games[games["total_pts"].isna() & (games["is_playoff"] == 0) & (games["season"] >= season_now)]
    wk = int(todo["week"].min()) if len(todo) else None
    teams_now = sorted(set(games.loc[games["season"] == season_now, "home_team"]) |
                       set(games.loc[games["season"] == season_now, "away_team"]))
    tg, cur = add_pregame_ratings(tg, teams_now, season_now)
    d, X = historical_table(games, tg)
    log(f"{len(d)} completed games with full pregame ratings ({d['season'].min()}-{d['season'].max()})")

    # ---- honest out-of-sample check against the closing total
    r, summ = backtest(d, X)
    if summ is not None:
        s = summ.iloc[0]
        print(f"\nOUT-OF-SAMPLE {TEST_SEASONS}, {int(s['games'])} games, mean miss in points: model {s['mae_model']:.3f} | "
              f"closing total {s['mae_closing_total']:.3f} | 50/50 blend {s['mae_blend_50_50']:.3f} | league average {s['mae_league_avg']:.3f}")
        print(f"  correlation of (model - closing total) with (actual - closing total): {s['corr_edge_vs_miss']:+.3f} "
              f"(0 = our disagreement with the market carries no information)")
        rec = over_under_record(r["edge"], r["total_pts"], r["mkt_total_close"])
        print("  over/under record when the model disagrees with the closing total (pushes excluded; need > 52.4% to beat -110):")
        print(rec.round(3).to_string(index=False))
        summ.to_csv(f"{BOARD_DIR}/totals_backtest_latest.csv", index=False)
        rec.to_csv(f"{BOARD_DIR}/totals_backtest_record_latest.csv", index=False)
        oinfo, orec = open_line_check(r)
        if oinfo is not None:
            print(f"\nOPENING-LINE CHECK, {oinfo['open_games']} games with an opening total (the angle that matters if you bet early):")
            print(f"  correlation of (model - opening total) with (closing - opening total): {oinfo['corr_edge_vs_move']:+.3f} "
                  f"(positive = the market tends to move toward the model)")
            print("  betting the OPENING total on the model's side (pushes excluded; need > 52.4% at -110), and how the line then moved:")
            print(orec.round(3).to_string(index=False))
            orec.to_csv(f"{BOARD_DIR}/totals_open_line_record_latest.csv", index=False)
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
    board["weather_known"] = weather_known.values
    board["market_total"], board["market_src"] = up["mkt_total_close"].values, "nflverse schedule line"
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
    board["edge_vs_market"] = board["fair_total"] - board["market_total"]            # + = model likes the over
    board["lean"] = np.where(board["edge_vs_market"].abs() < 0.05, "",
                             np.where(board["edge_vs_market"] > 0, "Over", "Under"))
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
