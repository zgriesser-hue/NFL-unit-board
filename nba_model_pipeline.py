"""NBA spread and totals model with a walk-forward backtest, built on free team game logs (data/nba_games.csv).

Ratings (all measured per 100 possessions, relative to the league, using only games played BEFORE each game):
  off  = points scored, adjusted for the opponent's defense;   def = points allowed, adjusted for the opponent's offense
  pace = possessions per game.
Each rating is a recency-weighted average that starts from last season's final rating (shrunk toward average) and shifts
to this season's games as they come in. Also in the model: rest days, back-to-backs, the league's scoring level.

Two ridge regressions (margin = home - away, and total) are trained on earlier seasons only and tested on later ones.
If data/nba_lines_hist.csv exists (date, home, away, home_spread, total; home_spread positive = home favored) the backtest also
compares against the market and shows the win rate when the model disagrees. Without it the test only measures accuracy.

Upcoming games and market lines come from boards/nba_lines.csv (the daily snapshot) and are written to boards/nba_board_latest.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NFL_ROOT", HERE)
DATA, BOARD = f"{ROOT}/data", f"{ROOT}/boards"
K_PRIOR = float(os.environ.get("NBA_K_PRIOR", "12"))      # games' worth of weight on last season's rating
DECAY = float(os.environ.get("NBA_DECAY", "0.97"))        # each older game counts this much less
CARRY = float(os.environ.get("NBA_CARRY", "0.7"))         # share of last season's rating kept for next season
ALPHA = float(os.environ.get("NBA_ALPHA", "30"))
N_TEST = int(os.environ.get("NBA_TEST_SEASONS", "3"))
FEATS = ["off_h", "def_h", "off_a", "def_a", "pace_h", "pace_a", "rest_h", "rest_a", "b2b_h", "b2b_a", "lg_ortg", "lg_pace"]
FEATS_T = [f for f in FEATS if f not in ("lg_ortg", "lg_pace")]       # the total model works relative to the league's scoring level
AVAIL = ["avail_h", "avail_a"]
P_DECAY = float(os.environ.get("NBA_P_DECAY", "0.97"))    # each older game counts this much less for a player
P_SHRINK = float(os.environ.get("NBA_P_SHRINK", "400"))    # minutes' worth of weight on the league-average prior for a player's rate
P_DELTA = float(os.environ.get("NBA_P_DELTA", "0.06"))     # unknown players are assumed this much below the league rate (bench level)
SPREAD_EDGES, TOTAL_EDGES = [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]

NAMES = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
         "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
         "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC",
         "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
         "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
         "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR",
         "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"}


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ data
def load_games(path):
    g = pd.read_csv(path, dtype={"game_id": str})
    g = g[g["home"].isin(set(NAMES.values())) & g["away"].isin(set(NAMES.values()))].copy()       # drop All-Star and exhibition teams
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    num = ["home_pts", "away_pts", "minutes"] + [f"{s}_{c}" for s in ("home", "away") for c in ("fga", "fta", "oreb", "tov")]
    g[num] = g[num].apply(pd.to_numeric, errors="coerce")
    for s in ("home", "away"):
        g[f"{s}_poss"] = g[f"{s}_fga"] + 0.44 * g[f"{s}_fta"] - g[f"{s}_oreb"] + g[f"{s}_tov"]
    ok = g["date"].notna() & g[["home_pts", "away_pts", "minutes", "home_poss", "away_poss"]].notna().all(axis=1) \
        & g["home_poss"].between(60, 140) & g["away_poss"].between(60, 140)                  # missing or impossible box scores
    if (~ok).any():
        bad = g[~ok].groupby("season").size().to_dict()
        log(f"dropped {int((~ok).sum())} games with missing or impossible box-score numbers (by season): {bad}")
    g = g[ok].sort_values(["date", "game_id"]).reset_index(drop=True)
    scale = 240.0 / g["minutes"].clip(lower=200)                      # overtime games: put everything on a 48-minute basis
    g["poss"] = (g["home_poss"] + g["away_poss"]) / 2 * scale
    g["h_ortg"] = g["home_pts"] / (g["home_poss"] + g["away_poss"]) * 200      # points per 100 possessions
    g["a_ortg"] = g["away_pts"] / (g["home_poss"] + g["away_poss"]) * 200
    return g


# ------------------------------------------------------------------ ratings engine
class Ratings:
    """Pregame team ratings, updated one date at a time so no game ever sees its own result."""

    def __init__(self):
        self.t, self.prior, self.lg = {}, {}, {"ortg": 112.0, "pace": 99.0, "w": 0.0, "so": 0.0, "sp": 0.0}
        self.season = None

    def _team(self, tm):
        if tm not in self.t:
            p = self.prior.get(tm, (0.0, 0.0, 0.0))
            self.t[tm] = {"w": 0.0, "so": 0.0, "sd": 0.0, "sp": 0.0, "last": None, "p": p}
        return self.t[tm]

    def new_season(self, season):
        if self.season is not None:                                    # save each team's final rating as next season's prior
            for tm in list(self.t):
                o, d, p = self.get(tm)
                self.prior[tm] = (CARRY * o, CARRY * d, CARRY * p)
            if self.lg["w"] > 0:
                self.lg.update(ortg=self.lg["so"] / self.lg["w"], pace=self.lg["sp"] / self.lg["w"])
        self.t, self.season = {}, season
        self.lg.update(w=0.0, so=0.0, sp=0.0)

    def get(self, tm):
        s = self._team(tm)
        den = s["w"] + K_PRIOR
        o, d, p = s["p"]
        return ((s["so"] + K_PRIOR * o) / den, (s["sd"] + K_PRIOR * d) / den, (s["sp"] + K_PRIOR * p) / den)

    def features(self, home, away, date):
        oh, dh, ph = self.get(home)
        oa, da, pa = self.get(away)
        rest = lambda tm: 5.0 if self._team(tm)["last"] is None else float(min((date - self._team(tm)["last"]).days, 5))
        rh, ra = rest(home), rest(away)
        return {"off_h": oh, "def_h": dh, "off_a": oa, "def_a": da, "pace_h": ph, "pace_a": pa, "rest_h": rh, "rest_a": ra,
                "b2b_h": float(rh <= 1), "b2b_a": float(ra <= 1), "lg_ortg": self.lg["ortg"], "lg_pace": self.lg["pace"]}

    def update(self, rows):
        """rows: games played on one date (a DataFrame). Ratings are relative to the league, per 100 possessions."""
        pre = {}
        for r in rows.itertuples():
            pre[r.game_id] = (self.get(r.home), self.get(r.away))
        for r in rows.itertuples():
            (oh, dh, ph), (oa, da, pa) = pre[r.game_id]
            ho, ao = r.h_ortg - self.lg["ortg"], r.a_ortg - self.lg["ortg"]
            pace_rel = r.poss - self.lg["pace"]
            for tm, off_rel, opp_def, allowed_rel, opp_off in ((r.home, ho, da, ao, oa), (r.away, ao, dh, ho, oh)):
                s = self._team(tm)
                for k in ("w", "so", "sd", "sp"):
                    s[k] *= DECAY
                s["w"] += 1.0
                s["so"] += off_rel - opp_def                           # scored this much more than average, beyond the defense faced
                s["sd"] += allowed_rel - opp_off                       # allowed this much more than average, beyond the offense faced
                s["sp"] += pace_rel
                s["last"] = r.date
            self.lg["w"] += 1.0
            self.lg["so"] += (r.h_ortg + r.a_ortg) / 2
            self.lg["sp"] += r.poss
        if self.lg["w"] >= 40:                                          # blend the league level in once there are enough games
            k = min(self.lg["w"] / 200.0, 1.0)
            self.lg["ortg"] = (1 - k) * self.lg["ortg"] + k * self.lg["so"] / self.lg["w"]
            self.lg["pace"] = (1 - k) * self.lg["pace"] + k * self.lg["sp"] / self.lg["w"]


def history_table(g):
    eng, rows = Ratings(), []
    for date, day in g.groupby("date", sort=True):
        if day["season"].iloc[0] != eng.season:
            eng.new_season(int(day["season"].iloc[0]))
        for r in day.itertuples():
            f = eng.features(r.home, r.away, date)
            f.update(game_id=r.game_id, season=int(r.season), date=date, home=r.home, away=r.away,
                     margin=r.home_pts - r.away_pts, total=r.home_pts + r.away_pts,
                     games_played=eng._team(r.home)["w"] + eng._team(r.away)["w"])
            rows.append(f)
        eng.update(day)
    return pd.DataFrame(rows), eng


# ------------------------------------------------------------------ models
class Ridged:
    def fit(self, X, y):
        self.mu, self.sd = X.mean(), X.std().replace(0, 1.0)
        self.m = Ridge(alpha=ALPHA).fit(((X - self.mu) / self.sd).values, y.values)
        self.coef = pd.Series(self.m.coef_, index=X.columns)
        return self

    def predict(self, X):
        return pd.Series(self.m.predict(((X - self.mu) / self.sd).values), index=X.index)


def lg_total(X):
    """The league's expected total at this point (points per 100 possessions x possessions x 2 teams)."""
    return X["lg_ortg"] * X["lg_pace"] / 50.0


class TotalRidge(Ridged):
    """Predicts how far a game's total sits from the league's scoring level, so a season with a new scoring level
    (a pandemic year, a rule change) does not throw the model off. Fit and predict take the full FEATS frame."""

    def __init__(self, cols=None):
        self.cols = cols or FEATS_T

    def fit(self, X, y):
        return super().fit(X[self.cols], y - lg_total(X))

    def predict(self, X):
        return super().predict(X[self.cols]) + lg_total(X)


def win_record(edge, actual_vs_line, thresholds):
    """edge > 0 = model says the side that wins when actual_vs_line > 0. Pushes dropped."""
    rows = []
    for t in thresholds:
        m = (edge.abs() >= t) & (actual_vs_line != 0)
        n = int(m.sum())
        win = (np.sign(edge[m]) == np.sign(actual_vs_line[m]))
        rows.append({"edge_at_least": t, "games": n, "win_pct": float(win.mean()) if n else np.nan})
    return pd.DataFrame(rows)


def seasons_with_lines(d):
    """Seasons where data/nba_lines_hist.csv covers a good share of games (so the backtest can be graded against the market)."""
    path = f"{DATA}/nba_lines_hist.csv"
    if not os.path.exists(path):
        return None
    h = pd.read_csv(path)
    h["date"] = pd.to_datetime(h["date"])
    m = d[["season", "date", "home", "away"]].merge(h[["date", "home", "away"]].drop_duplicates(), on=["date", "home", "away"], how="inner")
    cnt = m.groupby("season").size()
    return {int(s) for s, n in cnt.items() if n >= 500}


def backtest(d):
    seasons = sorted(d["season"].unique())
    full = [s for s in seasons if (d["season"] == s).sum() >= 1100]
    have = seasons_with_lines(d)
    if have:                                                    # test on seasons the saved lines cover, so the market comparison works
        full = [s for s in full if s in have]
    test = [int(s) for s in full if s >= seasons[0] + 2][-N_TEST:]                   # first seasons are only warm-up for the ratings
    out = []
    for s in test:
        tr = (d["season"] < s) & (d["season"] > seasons[0])                      # skip the very first season (ratings still cold)
        te = d["season"] == s
        if tr.sum() < 1000:
            continue
        t = d[te][["game_id", "season", "date", "home", "away", "margin", "total", "games_played"]].copy()
        t["p_margin"] = Ridged().fit(d.loc[tr, FEATS], d.loc[tr, "margin"]).predict(d.loc[te, FEATS]).values
        t["p_total"] = TotalRidge().fit(d.loc[tr, FEATS], d.loc[tr, "total"]).predict(d.loc[te, FEATS]).values
        for tgt in ("margin", "total"):
            t[f"base_{tgt}"] = float(d.loc[tr, tgt].mean())
        out.append(t)
    return (pd.concat(out) if out else None), test


def report(r, test):
    mae = lambda a, b: float((a - b).abs().mean())
    log(f"\nWALK-FORWARD TEST on {len(r)} games, seasons {test} (each trained only on earlier seasons). Mean miss in points, lower is better:")
    log(f"  margin: model {mae(r['p_margin'], r['margin']):.3f} | home-court-only baseline {mae(r['base_margin'], r['margin']):.3f}")
    log(f"  total:  model {mae(r['p_total'], r['total']):.3f} | league-average baseline {mae(r['base_total'], r['total']):.3f}")
    late = r["games_played"] >= 40
    log(f"  after the first ~20 games per team: margin {mae(r.loc[late, 'p_margin'], r.loc[late, 'margin']):.3f}, "
        f"total {mae(r.loc[late, 'p_total'], r.loc[late, 'total']):.3f}")
    path = f"{DATA}/nba_lines_hist.csv"
    if not os.path.exists(path):
        log("\n  (no data/nba_lines_hist.csv: add historical closing spreads and totals to compare with the market. "
            "Columns: date, home, away, home_spread (positive = home favored), total.)")
        return
    h = pd.read_csv(path)
    h["date"] = pd.to_datetime(h["date"])
    m = r.merge(h[["date", "home", "away", "home_spread", "total"]].rename(columns={"total": "mkt_total"}), on=["date", "home", "away"], how="inner")
    if len(m) < 300:
        log(f"\n  (only {len(m)} games matched data/nba_lines_hist.csv; check the date and team abbreviations)")
        return
    log(f"\nVS THE MARKET, {len(m)} games with closing lines:")
    log(f"  spread: model {mae(m['p_margin'], m['margin']):.3f} | market {mae(m['home_spread'], m['margin']):.3f} | 50/50 blend "
        f"{mae(0.5 * m['p_margin'] + 0.5 * m['home_spread'], m['margin']):.3f}")
    log(f"  total:  model {mae(m['p_total'], m['total']):.3f} | market {mae(m['mkt_total'], m['total']):.3f} | 50/50 blend "
        f"{mae(0.5 * m['p_total'] + 0.5 * m['mkt_total'], m['total']):.3f}")
    se, te_ = m["p_margin"] - m["home_spread"], m["p_total"] - m["mkt_total"]
    log(f"  corr of (model - market) with the market's miss: spread {np.corrcoef(se, m['margin'] - m['home_spread'])[0, 1]:+.3f}, "
        f"total {np.corrcoef(te_, m['total'] - m['mkt_total'])[0, 1]:+.3f}  (0 = no information beyond the line)")
    log("  spread: win rate betting the model's side against the closing line (need > 52.4% at -110):")
    log(win_record(se, m["margin"] - m["home_spread"], SPREAD_EDGES).round(3).to_string(index=False))
    log("  total: win rate betting the model's side (Over/Under) against the closing total:")
    log(win_record(te_, m["total"] - m["mkt_total"], TOTAL_EDGES).round(3).to_string(index=False))


def total_diagnostic(d, test):
    """Why is the total model off? Re-run the walk-forward test with different feature sets and compare with the market."""
    path = f"{DATA}/nba_lines_hist.csv"
    if not os.path.exists(path) or not test:
        return
    h = pd.read_csv(path)
    h["date"] = pd.to_datetime(h["date"])
    variants = {"all features, raw total (the old way)": ("raw", FEATS),
                "league-relative total (current)": ("rel", FEATS_T),
                "  ...without pace": ("rel", [f for f in FEATS_T if not f.startswith("pace")]),
                "  ...without pace or rest": ("rel", [f for f in FEATS_T if not f.startswith(("pace", "rest", "b2b"))]),
                "  ...offense and defense only": ("rel", ["off_h", "def_h", "off_a", "def_a"])}
    rows = []
    for name, (kind, cols) in variants.items():
        out = []
        for s in test:
            tr, te = (d["season"] < s) & (d["season"] > d["season"].min()), d["season"] == s
            if tr.sum() < 1000:
                continue
            t = d[te][["date", "home", "away", "total"]].copy()
            m = Ridged().fit(d.loc[tr, cols], d.loc[tr, "total"]) if kind == "raw" else TotalRidge(cols).fit(d.loc[tr, FEATS], d.loc[tr, "total"])
            t["p"] = m.predict(d.loc[te, cols] if kind == "raw" else d.loc[te, FEATS]).values
            t["base"] = float(d.loc[tr, "total"].mean())
            out.append(t)
        if out:
            r = pd.concat(out).merge(h[["date", "home", "away", "total"]].rename(columns={"total": "mkt"}), on=["date", "home", "away"])
            rows.append({"total model": name, "games": len(r), "model_miss": float((r["p"] - r["total"]).abs().mean()),
                         "market_miss": float((r["mkt"] - r["total"]).abs().mean()), "league_avg_miss": float((r["base"] - r["total"]).abs().mean())})
    if rows:
        log("\nTOTAL MODEL DIAGNOSTIC (mean miss on games with closing totals; the market is the number to beat):")
        log(pd.DataFrame(rows).round(3).to_string(index=False))


# ------------------------------------------------------------------ availability (who played)
def availability(players):
    """Lineup strength for each team in each game, and how far it is from that team's usual lineup (all pregame information
    about the players; WHO ACTUALLY PLAYED is known only after the fact, so this is a ceiling test, not a betting signal).

    A player's value = (game score per minute, shrunk toward the league rate) - league rate. Lineup strength L = sum over the
    players who played of expected minutes x value. avail = L - the team's recent average L (so a missing star is negative).
    Returns a frame with game_id, team, L, avail."""
    p = players[players["team"].isin(set(NAMES.values()))].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["date", "gid"])
    st = {}                      # player -> [decayed minutes, decayed game score, decayed games played, decayed minutes (played games)]
    ref = {}                     # team -> [decayed sum of L, decayed count]
    lg = [1.0, 1.0]              # league minutes and game score (all players, slowly decayed): league rate = lg[1] / lg[0]
    out = []
    for date, day in p.groupby("date", sort=True):
        lg_rate = lg[1] / lg[0]
        todo = []
        for (gid, team), g in day.groupby(["gid", "team"], sort=False):
            act = g[g["min"] > 0]
            L = 0.0
            for r in act.itertuples():
                m, gs, n, _ = st.get(r.pid, (0.0, 0.0, 0.0, 0.0))
                rate = (gs + P_SHRINK * (lg_rate - P_DELTA)) / (m + P_SHRINK)
                exp_min = (m / n) if n > 0.5 else 10.0
                L += exp_min * (rate - lg_rate)
            num, den = ref.get(team, (0.0, 0.0))
            out.append((gid, team, L, L - num / den if den > 0 else 0.0))
            todo.append((team, L, act))
        for team, L, act in todo:                                       # update only after the whole day is scored
            num, den = ref.get(team, (0.0, 0.0))
            ref[team] = (num * P_DECAY + L, den * P_DECAY + 1.0)
            for r in act.itertuples():
                m, gs, n, _ = st.get(r.pid, (0.0, 0.0, 0.0, 0.0))
                st[r.pid] = (m * P_DECAY + r.min, gs * P_DECAY + r.gmsc, n * P_DECAY + 1.0, 0.0)
                lg = [lg[0] * 0.99999 + r.min, lg[1] * 0.99999 + r.gmsc]
    return pd.DataFrame(out, columns=["game_id", "team", "L", "avail"])


def load_players():
    for name, reader in (("nba_players.parquet", pd.read_parquet), ("nba_players.csv.gz", pd.read_csv)):
        if os.path.exists(f"{DATA}/{name}"):
            return reader(f"{DATA}/{name}", dtype={"gid": str, "pid": str}) if name.endswith("gz") else reader(f"{DATA}/{name}")
    return None


def add_availability(d, players):
    a = availability(players).drop_duplicates(["game_id", "team"])
    d = d.merge(a.rename(columns={"team": "home", "L": "L_h", "avail": "avail_h"}), on=["game_id", "home"], how="left")
    d = d.merge(a.rename(columns={"team": "away", "L": "L_a", "avail": "avail_a"}), on=["game_id", "away"], how="left")
    miss = d["avail_h"].isna() | d["avail_a"].isna()
    log(f"availability computed for {int((~miss).sum())} of {len(d)} games ({int(miss.sum())} without player data count as average)")
    d[AVAIL + ["L_h", "L_a"]] = d[AVAIL + ["L_h", "L_a"]].fillna(0.0)
    return d


def availability_test(d, test):
    """The ceiling check: do the models improve if we know who played? Walk-forward, compared with the market where lines exist."""
    path = f"{DATA}/nba_lines_hist.csv"
    h = None
    if os.path.exists(path):
        h = pd.read_csv(path)
        h["date"] = pd.to_datetime(h["date"])
    F2, FT2 = FEATS + AVAIL, FEATS_T + AVAIL
    rows = []
    for s in test:
        tr, te = (d["season"] < s) & (d["season"] > d["season"].min()), d["season"] == s
        if tr.sum() < 1000:
            continue
        t = d[te][["date", "home", "away", "margin", "total"]].copy()
        t["m0"] = Ridged().fit(d.loc[tr, FEATS], d.loc[tr, "margin"]).predict(d.loc[te, FEATS]).values
        t["m1"] = Ridged().fit(d.loc[tr, F2], d.loc[tr, "margin"]).predict(d.loc[te, F2]).values
        t["t0"] = TotalRidge().fit(d.loc[tr, FEATS], d.loc[tr, "total"]).predict(d.loc[te, FEATS]).values
        t["t1"] = TotalRidge(FT2).fit(d.loc[tr, F2], d.loc[tr, "total"]).predict(d.loc[te, F2]).values
        rows.append(t)
    if not rows:
        return
    r = pd.concat(rows)
    mae = lambda a, b: float((a - b).abs().mean())
    log(f"\nAVAILABILITY CEILING TEST, {len(r)} games (the model is told who actually played, which Monday's board could not know):")
    log(f"  margin miss: ratings only {mae(r['m0'], r['margin']):.3f} -> with availability {mae(r['m1'], r['margin']):.3f}")
    log(f"  total miss:  ratings only {mae(r['t0'], r['total']):.3f} -> with availability {mae(r['t1'], r['total']):.3f}")
    if h is not None:
        m = r.merge(h[["date", "home", "away", "home_spread", "total"]].rename(columns={"total": "mkt_total"}), on=["date", "home", "away"])
        if len(m) >= 300:
            log(f"  against the closing lines ({len(m)} games): spread model {mae(m['m1'], m['margin']):.3f} vs market {mae(m['home_spread'], m['margin']):.3f}; "
                f"total model {mae(m['t1'], m['total']):.3f} vs market {mae(m['mkt_total'], m['total']):.3f}")
            se, te_ = m["m1"] - m["home_spread"], m["t1"] - m["mkt_total"]
            log(f"  corr of (model - market) with the market's miss: spread {np.corrcoef(se, m['margin'] - m['home_spread'])[0, 1]:+.3f}, "
                f"total {np.corrcoef(te_, m['total'] - m['mkt_total'])[0, 1]:+.3f}")
            log("  spread win rate betting the availability model's side (need > 52.4%):")
            log(win_record(se, m["margin"] - m["home_spread"], SPREAD_EDGES).round(3).to_string(index=False))
            log("  total win rate:")
            log(win_record(te_, m["total"] - m["mkt_total"], TOTAL_EDGES).round(3).to_string(index=False))
    tr_all = d["season"] > d["season"].min()
    mm = Ridged().fit(d.loc[tr_all, F2], d.loc[tr_all, "margin"])
    tt = TotalRidge(FT2).fit(d.loc[tr_all, F2], d.loc[tr_all, "total"])
    log("  weights for the availability terms (points per 1 SD): margin " + ", ".join(f"{k} {mm.coef[k]:+.2f}" for k in AVAIL)
        + " | total " + ", ".join(f"{k} {tt.coef[k]:+.2f}" for k in AVAIL))


# ------------------------------------------------------------------ board
def build_board(eng, mdl_m, mdl_t):
    path = f"{BOARD}/nba_lines.csv"
    if not os.path.exists(path):
        log("\n(no boards/nba_lines.csv yet: the daily line snapshot has not saved any games)")
        return
    s = pd.read_csv(path)
    s["kick"] = pd.to_datetime(s["kickoff_utc"], utc=True)
    s = s[s["kick"] > pd.Timestamp.now(tz="UTC")].sort_values("captured_utc").groupby(["home_team", "away_team", "kickoff_utc"], as_index=False).tail(1)
    s["home"], s["away"] = s["home_team"].map(NAMES), s["away_team"].map(NAMES)
    s = s.dropna(subset=["home", "away"])
    if s.empty:
        log("\n(no upcoming NBA games in the line snapshot)")
        return
    rows = []
    for r in s.itertuples():
        date = r.kick.tz_convert("America/New_York").tz_localize(None).normalize()
        rows.append(eng.features(r.home, r.away, date))
    X = pd.DataFrame(rows, index=s.index)[FEATS]
    b = s[["kickoff_utc", "home", "away", "home_spread", "total", "spread_books", "total_books", "best_home_point", "best_home_price",
           "best_home_book", "best_away_point", "best_away_price", "best_away_book", "best_over_line", "best_over_price",
           "best_over_book", "best_under_line", "best_under_price", "best_under_book"]].copy()
    b = b.rename(columns={"home_spread": "market_spread", "total": "market_total"})
    b["model_spread"], b["model_total"] = mdl_m.predict(X).values, mdl_t.predict(X).values
    b["edge_spread"] = b["model_spread"] - b["market_spread"]            # + = model likes the home team
    b["edge_total"] = b["model_total"] - b["market_total"]               # + = model likes the Over
    b["spread_lean"] = np.where(b["edge_spread"].abs() < 1, "", np.where(b["edge_spread"] > 0, b["home"] + " " + (-b["market_spread"]).map("{:+g}".format),
                                                                         b["away"] + " " + b["market_spread"].map("{:+g}".format)))
    b["total_lean"] = np.where(b["edge_total"].abs() < 2, "", np.where(b["edge_total"] > 0, "Over", "Under"))
    b["built_at_utc"] = pd.Timestamp.now(tz="UTC")
    b = b.sort_values("kickoff_utc").reset_index(drop=True)
    b.to_csv(f"{BOARD}/nba_board_latest.csv", index=False)
    pd.set_option("display.width", 220)
    log("\nNBA BOARD (edge = model minus market; spread edge + = home, total edge + = Over; leans shown at 1+ point spread / 2+ point total):")
    log(b[["away", "home", "market_spread", "model_spread", "edge_spread", "spread_lean", "market_total", "model_total", "edge_total", "total_lean"]]
        .round(1).to_string(index=False))


def main():
    path = f"{DATA}/nba_games.csv"
    if not os.path.exists(path):
        log(f"No {path}. Run nba_fetch_games.py first.")
        return 1
    g = load_games(path)
    d, eng = history_table(g)
    log(f"{len(d)} games with pregame ratings ({d['season'].min()}-{d['season'].max()}, season = start year)")
    r, test = backtest(d)
    if r is None:
        log("Not enough history for the walk-forward test (need 3+ full seasons).")
    else:
        report(r, test)
        total_diagnostic(d, test)
        try:
            pl = load_players()
            if pl is None:
                log("\n(no data/nba_players.parquet: run nba_fetch_players.py to test player availability)")
            else:
                availability_test(add_availability(d, pl), test)
        except Exception as e:
            log(f"(availability test skipped: {str(e)[:200]})")
        os.makedirs(BOARD, exist_ok=True)
        r.to_csv(f"{BOARD}/nba_backtest_latest.csv", index=False)
    tr = d["season"] > d["season"].min()
    mdl_m, mdl_t = Ridged().fit(d.loc[tr, FEATS], d.loc[tr, "margin"]), TotalRidge().fit(d.loc[tr, FEATS], d.loc[tr, "total"])
    pd.DataFrame({"margin_coef": mdl_m.coef, "total_coef": mdl_t.coef.reindex(FEATS)}).rename_axis("feature").reset_index().to_csv(
        f"{BOARD}/nba_model_weights_latest.csv", index=False)
    log("\nMODEL WEIGHTS (points per 1 standard deviation of the feature):")
    log(pd.DataFrame({"margin": mdl_m.coef, "total (vs league level)": mdl_t.coef.reindex(FEATS)}).round(2).to_string())
    build_board(eng, mdl_m, mdl_t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
