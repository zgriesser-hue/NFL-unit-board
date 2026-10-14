"""College football spread and totals model with a walk-forward backtest against the market (data/cfb_games.csv from cfb_fetch.py).

Team ratings (points per game, relative to the league, using only games played BEFORE each game):
  off = points scored beyond what the opponent's defense normally allows;  def = points allowed beyond what the opponent's offense normally scores.
Each rating starts from last season's final rating (shrunk toward average, since rosters turn over) and moves to this season's games as they come in.
Two ridge regressions (margin = home - away, and total) are trained on earlier seasons only and tested on the last three seasons, against the
closing lines and the opening lines. The rating settings are tuned on EARLIER seasons than the ones graded, separately for margin and total.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NFL_ROOT", HERE)
DATA, BOARD = f"{ROOT}/data", f"{ROOT}/boards"
HFA = 1.2                                    # points each side's score is nudged for home field, so ratings are not biased by who hosted
N_TEST = int(os.environ.get("CFB_TEST_SEASONS", "3"))
DEFAULT = {"K_PRIOR": 6.0, "DECAY": 0.95, "CARRY": 0.6, "ALPHA": 30.0}
GRID = {"K_PRIOR": [2, 4, 8, 14], "DECAY": [0.8, 0.9, 0.95, 1.0], "CARRY": [0.2, 0.4, 0.6, 0.8], "ALPHA": [3, 30, 100, 300]}
FEATS = ["off_h", "def_h", "off_a", "def_a", "rest_h", "rest_a", "neutral", "lg_pts"]
FEATS_T = [f for f in FEATS if f != "lg_pts"]
SPREAD_EDGES, TOTAL_EDGES = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 6.0]


def log(m):
    print(m, flush=True)


def load(path):
    g = pd.read_csv(path)
    d = pd.to_datetime(g["date"], utc=True, errors="coerce") - pd.Timedelta(hours=6)       # late games keep their US date
    g["date"] = d.dt.tz_localize(None).dt.normalize()
    for c in ("home_pts", "away_pts", "spread_close", "spread_open", "total_close", "total_open"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    for s in ("home", "away"):
        cls = g[f"{s}_class"].astype(str).str.lower()
        g[f"{s}_fbs"] = cls.str.contains("fbs") | cls.isin(["nan", "none"])        # when the site gives no classification, keep the game
    g = g[g["home_fbs"] & g["away_fbs"] & g["date"].notna()].copy()
    g["neutral"] = g["neutral"].astype(str).str.lower().isin(["true", "1"]).astype(float)
    g["played"] = g["home_pts"].notna() & g["away_pts"].notna()
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


class Ratings:
    def __init__(self, p):
        self.p, self.t, self.prior, self.season, self.lg = p, {}, {}, None, {"pts": 29.0, "w": 0.0, "s": 0.0}

    def _team(self, tm):
        if tm not in self.t:
            self.t[tm] = {"w": 0.0, "so": 0.0, "sd": 0.0, "last": None, "p": self.prior.get(tm, (0.0, 0.0))}
        return self.t[tm]

    def new_season(self, season):
        if self.season is not None:
            for tm in list(self.t):
                o, d = self.get(tm)
                self.prior[tm] = (self.p["CARRY"] * o, self.p["CARRY"] * d)
            if self.lg["w"] > 0:
                self.lg["pts"] = self.lg["s"] / self.lg["w"]
        self.t, self.season = {}, season
        self.lg.update(w=0.0, s=0.0)

    def get(self, tm):
        s = self._team(tm)
        den = s["w"] + self.p["K_PRIOR"]
        return ((s["so"] + self.p["K_PRIOR"] * s["p"][0]) / den, (s["sd"] + self.p["K_PRIOR"] * s["p"][1]) / den)

    def features(self, home, away, date, neutral):
        oh, dh = self.get(home)
        oa, da = self.get(away)
        rest = lambda tm: 7.0 if self._team(tm)["last"] is None else float(min((date - self._team(tm)["last"]).days, 14))
        lg = self.lg["pts"] if self.lg["w"] < 30 else (self.lg["w"] * self.lg["s"] / self.lg["w"] + 30 * self.lg["pts"]) / (self.lg["w"] + 30)
        return {"off_h": oh, "def_h": dh, "off_a": oa, "def_a": da, "rest_h": rest(home), "rest_a": rest(away), "neutral": neutral, "lg_pts": lg}

    def update(self, rows):
        pre = {r.game_id: (self.get(r.home), self.get(r.away)) for r in rows.itertuples()}
        for r in rows.itertuples():
            (oh, dh), (oa, da) = pre[r.game_id]
            hf = 0.0 if r.neutral else HFA
            lg = self.lg["pts"]
            hp, ap = r.home_pts - hf - lg, r.away_pts + hf - lg
            for tm, scored, allowed, opp_def, opp_off in ((r.home, hp, ap, da, oa), (r.away, ap, hp, dh, oh)):
                s = self._team(tm)
                for k in ("w", "so", "sd"):
                    s[k] *= self.p["DECAY"]
                s["w"] += 1.0
                s["so"] += scored - opp_def
                s["sd"] += allowed - opp_off
                s["last"] = r.date
            self.lg["w"] += 1.0
            self.lg["s"] += (r.home_pts + r.away_pts) / 2


def history(g, p):
    eng, rows = Ratings(p), []
    for date, day in g[g["played"]].groupby("date", sort=True):
        if day["season"].iloc[0] != eng.season:
            eng.new_season(int(day["season"].iloc[0]))
        for r in day.itertuples():
            f = eng.features(r.home, r.away, date, r.neutral)
            f.update(game_id=r.game_id, season=int(r.season), date=date, home=r.home, away=r.away, margin=r.home_pts - r.away_pts,
                     total=r.home_pts + r.away_pts, spread_close=r.spread_close, spread_open=r.spread_open,
                     total_close=r.total_close, total_open=r.total_open, season_type=r.season_type)
            rows.append(f)
        eng.update(day)
    return pd.DataFrame(rows)


class Model:
    def __init__(self, cols, alpha, total):
        self.cols, self.alpha, self.total = cols, alpha, total

    def y(self, d):
        return d["total"] - 2 * d["lg_pts"] if self.total else d["margin"]

    def fit(self, d):
        X = d[self.cols]
        self.mu, self.sd = X.mean(), X.std().replace(0, 1.0)
        self.m = Ridge(alpha=self.alpha).fit(((X - self.mu) / self.sd).values, self.y(d).values)
        self.coef = pd.Series(self.m.coef_, index=self.cols)
        return self

    def predict(self, d):
        p = self.m.predict(((d[self.cols] - self.mu) / self.sd).values)
        return p + 2 * d["lg_pts"].values if self.total else p


def walk(d, seasons, alpha, total):
    out, first = [], d["season"].min()
    for s in seasons:
        tr, te = d[(d["season"] < s) & (d["season"] > first)], d[d["season"] == s]
        if len(tr) < 800 or te.empty:
            continue
        m = Model(FEATS_T if total else FEATS, alpha, total).fit(tr)
        t = te.copy()
        t["p"] = m.predict(te)
        out.append(t)
    return pd.concat(out) if out else None


def mae(r, total):
    return float((r["p"] - r["total" if total else "margin"]).abs().mean())


def tune(g, val, total):
    cur, t0 = dict(DEFAULT), time.time()
    best = None
    for k in GRID:
        sc = {}
        for v in GRID[k]:
            p = {**cur, k: float(v)}
            sc[v] = mae(walk(history(g, p), val, p["ALPHA"], total), total)
        v = min(sc, key=sc.get)
        cur[k], best = float(v), sc[v]
        edge = "  (EDGE OF THE RANGE TRIED)" if v in (GRID[k][0], GRID[k][-1]) else ""
        log(f"  [{'total' if total else 'margin'}] {k}: " + ", ".join(f"{a}={b:.3f}" for a, b in sc.items()) + f" -> {v}{edge}  ({time.time() - t0:.0f}s)")
    return cur, best


def win_record(edge, vs_line, thresholds):
    rows = []
    for t in thresholds:
        m = (edge.abs() >= t) & (vs_line != 0) & edge.notna() & vs_line.notna()
        n = int(m.sum())
        rows.append({"edge_at_least": t, "games": n, "win_pct": float((np.sign(edge[m]) == np.sign(vs_line[m])).mean()) if n else np.nan})
    return pd.DataFrame(rows)


def main():
    path = f"{DATA}/cfb_games.csv"
    if not os.path.exists(path):
        log(f"No {path}. Run cfb_fetch.py first.")
        return 1
    g = load(path)
    played = g[g["played"]]
    log(f"{len(played)} played FBS-vs-FBS games, seasons {int(played['season'].min())}-{int(played['season'].max())}; "
        f"{int(played['spread_close'].notna().sum())} with a closing spread, {int(played['total_close'].notna().sum())} with a closing total")
    # the site gives the home team's spread (negative = home favored); confirm the sign from the data
    ok = played.dropna(subset=["spread_close"])
    sign = 1.0 if np.corrcoef(-ok["spread_close"], ok["home_pts"] - ok["away_pts"])[0, 1] > 0 else -1.0
    log(f"market spread sign check: home spread is {'negative when the home team is favored (as expected)' if sign > 0 else 'POSITIVE when the home team is favored; flipped'}")
    for c in ("spread_close", "spread_open"):
        g[c] = g[c] * -sign                          # now: positive = home favored, same scale as a predicted margin
    d0 = history(g, DEFAULT)
    full = [int(s) for s in sorted(d0["season"].unique()) if (d0["season"] == s).sum() >= 600 and d0.loc[d0["season"] == s, "spread_close"].notna().mean() > 0.6]
    test = full[-N_TEST:]
    val = [s for s in full if s < test[0] and s >= int(d0["season"].min()) + 2]
    log(f"tuning on seasons {val}; grading on {test} (never used for tuning)")
    if not val or not test:
        log("Not enough seasons.")
        return 1

    log("\nSTEP 1: TUNING the rating settings (mean miss on the actual result over the tuning seasons, lower is better)")
    p_m, s_m = tune(g, val, False)
    p_t, s_t = tune(g, val, True)
    log(f"  margin settings: {p_m} -> {s_m:.3f}\n  total settings:  {p_t} -> {s_t:.3f}")

    log("\nSTEP 2: GRADE on the last seasons")
    dm, dt = history(g, p_m), history(g, p_t)
    rm, rt = walk(dm, test, p_m["ALPHA"], False), walk(dt, test, p_t["ALPHA"], True)
    r0m, r0t = walk(d0, test, DEFAULT["ALPHA"], False), walk(d0, test, DEFAULT["ALPHA"], True)
    for name, r, r0, tot, cl, op in (("SPREAD", rm, r0m, False, "spread_close", "spread_open"), ("TOTAL", rt, r0t, True, "total_close", "total_open")):
        y = "total" if tot else "margin"
        m = r[r[cl].notna()]
        log(f"\n{name}: {len(m)} games with a closing line. Mean miss: default settings {mae(r0.loc[m.index], tot):.3f} | tuned {mae(m, tot):.3f} | "
            f"closing line {float((m[cl] - m[y]).abs().mean()):.3f}"
            + (f" | opening line {float((m[op] - m[y]).abs().mean()):.3f}" if m[op].notna().any() else "")
            + f" | 50/50 blend with the close {float((0.5 * m['p'] + 0.5 * m[cl] - m[y]).abs().mean()):.3f}")
        e = m["p"] - m[cl]
        log(f"  corr of (model - closing line) with the line's miss: {np.corrcoef(e, m[y] - m[cl])[0, 1]:+.3f}   (0 = nothing beyond the line)")
        log(f"  win rate betting the model's side against the CLOSING line (need > 52.4%):")
        log(win_record(e, m[y] - m[cl], TOTAL_EDGES if tot else SPREAD_EDGES).round(3).to_string(index=False))
        mo = m[m[op].notna()]
        if len(mo) > 300:
            log(f"  against the OPENING line ({len(mo)} games):")
            log(win_record(mo["p"] - mo[op], mo[y] - mo[op], TOTAL_EDGES if tot else SPREAD_EDGES).round(3).to_string(index=False))
    tr = dm[dm["season"] > dm["season"].min()]
    log("\nMODEL WEIGHTS (points per 1 standard deviation of the feature):")
    log(pd.DataFrame({"margin": Model(FEATS, p_m["ALPHA"], False).fit(tr).coef,
                      "total": Model(FEATS_T, p_t["ALPHA"], True).fit(dt[dt['season'] > dt['season'].min()]).coef}).round(2).to_string())
    os.makedirs(BOARD, exist_ok=True)
    rm.drop(columns=[c for c in ("season_type",) if c in rm]).to_csv(f"{BOARD}/cfb_backtest_spread_latest.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
