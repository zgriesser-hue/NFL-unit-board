"""College football spread and totals model with a walk-forward backtest against the market (data from cfb_fetch.py / cfb_fetch_weather.py).

Team ratings use only games played BEFORE each game, relative to the league, opponent-adjusted:
  points   off / def  - points scored / allowed beyond what the opponent usually gives up
  efficiency eoff / edef - the same for expected points added per play (a cleaner measure of how good a team is than the score)
  pace     plays per game
Each starts from last season's final rating (shrunk) and moves to this season's games as they come in. Ridge regressions (margin and total)
are trained on earlier seasons only. Steps: tune the rating settings on earlier seasons; grade on the last seasons; test the extra information
(pace, efficiency ratings, weather) season by season; then compare how the model does on smaller schools vs SEC / Big Ten games.
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
HFA = 1.2
N_TEST = int(os.environ.get("CFB_TEST_SEASONS", "3"))
DEFAULT = {"K_PRIOR": 6.0, "DECAY": 0.95, "CARRY": 0.6, "ALPHA": 30.0}
GRID = {"K_PRIOR": [2, 4, 8, 14, 24], "DECAY": [0.8, 0.9, 0.95, 1.0], "CARRY": [0.2, 0.4, 0.6, 0.8, 0.9, 1.0], "ALPHA": [3, 30, 100, 300]}
SPREAD_EDGES, TOTAL_EDGES = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 6.0]
POWER_CONFS = ("SEC", "Big Ten", "Big 12", "ACC", "Pac-12")
M_BASE = ["off_h", "def_h", "off_a", "def_a", "rest_h", "rest_a", "neutral", "lg_pts"]
EFF = ["eoff_h", "edef_h", "eoff_a", "edef_a"]
PACE = ["pace_h", "pace_a"]
WX = ["wind", "cold", "wet", "indoor"]
T_BASE = [c for c in M_BASE if c != "lg_pts"]
MARGIN_VARIANTS = {"points ratings only": M_BASE, "+ efficiency ratings": M_BASE + EFF}
TOTAL_VARIANTS = {"points ratings only": T_BASE, "+ pace": T_BASE + PACE, "+ efficiency ratings": T_BASE + EFF, "+ weather": T_BASE + WX,
                  "+ all four": T_BASE + PACE + EFF + WX}


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ data
def conf_tier(conf, team):
    c = str(conf)
    return "P" if (any(c == p or c.startswith(p) for p in POWER_CONFS) or team == "Notre Dame") else "G"


def load(root_data):
    g = pd.read_csv(f"{root_data}/cfb_games.csv")
    d = pd.to_datetime(g["date"], utc=True, errors="coerce") - pd.Timedelta(hours=6)
    g["date"] = d.dt.tz_localize(None).dt.normalize()
    for c in ("home_pts", "away_pts", "spread_close", "spread_open", "total_close", "total_open", "n_books"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    for s in ("home", "away"):
        cls = g[f"{s}_class"].astype(str).str.lower()
        g[f"{s}_fbs"] = cls.str.contains("fbs") | cls.isin(["nan", "none"])
    g = g[g["home_fbs"] & g["away_fbs"] & g["date"].notna()].copy()
    g["neutral"] = g["neutral"].astype(str).str.lower().isin(["true", "1"]).astype(float)
    g["played"] = g["home_pts"].notna() & g["away_pts"].notna()
    for s in ("home", "away"):
        conf = g[f"{s}_conf"] if f"{s}_conf" in g else pd.Series("", index=g.index)
        g[f"{s}_tier"] = [conf_tier(c, t) for c, t in zip(conf, g[s])]
        g[f"{s}_secb1g"] = conf.astype(str).isin(["SEC", "Big Ten"])
    g["matchup"] = np.where((g["home_tier"] == "P") & (g["away_tier"] == "P"), "power vs power",
                            np.where((g["home_tier"] == "G") & (g["away_tier"] == "G"), "smaller vs smaller", "power vs smaller"))
    g["secb1g"] = np.where(g["home_secb1g"] | g["away_secb1g"], "SEC / Big Ten involved", "neither")
    for c in ("h_plays", "a_plays", "h_oppa", "a_oppa"):
        g[c] = np.nan
    sp = f"{root_data}/cfb_stats.csv"
    if os.path.exists(sp):
        st = pd.read_csv(sp).dropna(subset=["game_id", "team"]).drop_duplicates(["game_id", "team"])
        st["game_id"] = pd.to_numeric(st["game_id"], errors="coerce")
        for s, pre in (("home", "h"), ("away", "a")):
            m = g[["game_id", s]].merge(st.rename(columns={"team": s}), on=["game_id", s], how="left")
            g[f"{pre}_plays"], g[f"{pre}_oppa"] = m["plays"].values, m["o_ppa"].values
    for c in ("temp_f", "wind_mph", "precip_in", "indoor"):
        g[c] = np.nan
    wp = f"{root_data}/cfb_weather.csv"
    if os.path.exists(wp):
        w = pd.read_csv(wp).drop_duplicates("game_id")
        m = g[["game_id"]].merge(w, on="game_id", how="left")
        for c in ("temp_f", "wind_mph", "precip_in", "indoor"):
            g[c] = m[c].values
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ ratings
class RunMean:
    def __init__(self, prior):
        self.prior, self.w, self.s = prior, 0.0, 0.0

    def val(self):
        return self.prior if self.w < 30 else (self.s + 30 * self.prior) / (self.w + 30)

    def add(self, x):
        self.w, self.s = self.w + 1.0, self.s + x

    def new_season(self):
        if self.w > 0:
            self.prior = self.s / self.w
        self.w = self.s = 0.0


class Ratings:
    def __init__(self, p):
        self.p, self.t, self.prior, self.season = p, {}, {}, None
        self.lg = {"pts": RunMean(29.0), "ppa": RunMean(0.0), "plays": RunMean(70.0)}

    def _team(self, tm):
        if tm not in self.t:
            self.t[tm] = {"w": 0.0, "so": 0.0, "sd": 0.0, "ew": 0.0, "eo": 0.0, "ed": 0.0, "pw": 0.0, "ps": 0.0, "last": None,
                          "p": self.prior.get(tm, (0.0,) * 5)}
        return self.t[tm]

    def new_season(self, season):
        if self.season is not None:
            c = self.p["CARRY"]
            for tm in list(self.t):
                self.prior[tm] = tuple(c * x for x in self.get(tm))
            for m in self.lg.values():
                m.new_season()
        self.t, self.season = {}, season

    def get(self, tm):
        s, K = self._team(tm), self.p["K_PRIOR"]
        p = s["p"]
        return ((s["so"] + K * p[0]) / (s["w"] + K), (s["sd"] + K * p[1]) / (s["w"] + K),
                (s["eo"] + K * p[2]) / (s["ew"] + K), (s["ed"] + K * p[3]) / (s["ew"] + K), (s["ps"] + K * p[4]) / (s["pw"] + K))

    def features(self, home, away, date, neutral):
        h, a = self.get(home), self.get(away)
        rest = lambda tm: 7.0 if self._team(tm)["last"] is None else float(min((date - self._team(tm)["last"]).days, 14))
        return {"off_h": h[0], "def_h": h[1], "off_a": a[0], "def_a": a[1], "eoff_h": h[2], "edef_h": h[3], "eoff_a": a[2], "edef_a": a[3],
                "pace_h": h[4], "pace_a": a[4], "rest_h": rest(home), "rest_a": rest(away), "neutral": neutral, "lg_pts": self.lg["pts"].val()}

    def update(self, rows):
        pre = {r.game_id: (self.get(r.home), self.get(r.away)) for r in rows.itertuples()}
        D = self.p["DECAY"]
        lg_pts, lg_ppa, lg_plays = self.lg["pts"].val(), self.lg["ppa"].val(), self.lg["plays"].val()
        for r in rows.itertuples():
            h, a = pre[r.game_id]
            hf = 0.0 if r.neutral else HFA
            hp, ap = r.home_pts - hf - lg_pts, r.away_pts + hf - lg_pts
            has_e = not (np.isnan(r.h_oppa) or np.isnan(r.a_oppa))
            has_p = not (np.isnan(r.h_plays) or np.isnan(r.a_plays))
            he, ae = (r.h_oppa - lg_ppa, r.a_oppa - lg_ppa) if has_e else (0.0, 0.0)
            for tm, scored, allowed, opp, me, own_e, opp_e, plays in ((r.home, hp, ap, a, h, he, ae, r.h_plays), (r.away, ap, hp, h, a, ae, he, r.a_plays)):
                s = self._team(tm)
                for k in ("w", "so", "sd"):
                    s[k] *= D
                s["w"] += 1.0
                s["so"] += scored - opp[1]
                s["sd"] += allowed - opp[0]
                s["last"] = r.date
                if has_e:
                    for k in ("ew", "eo", "ed"):
                        s[k] *= D
                    s["ew"] += 1.0
                    s["eo"] += own_e - opp[3]
                    s["ed"] += opp_e - opp[2]
                if has_p:
                    s["pw"], s["ps"] = s["pw"] * D + 1.0, s["ps"] * D + (plays - lg_plays)
            self.lg["pts"].add((r.home_pts + r.away_pts) / 2)
            if has_e:
                self.lg["ppa"].add((r.h_oppa + r.a_oppa) / 2)
            if has_p:
                self.lg["plays"].add((r.h_plays + r.a_plays) / 2)


def history(g, p, engine=False):
    eng, rows = Ratings(p), []
    for date, day in g[g["played"]].groupby("date", sort=True):
        if day["season"].iloc[0] != eng.season:
            eng.new_season(int(day["season"].iloc[0]))
        for r in day.itertuples():
            f = eng.features(r.home, r.away, date, r.neutral)
            ind = 1.0 if r.indoor == 1 else 0.0
            wind = 0.0 if (ind or np.isnan(r.wind_mph)) else float(np.clip(r.wind_mph - 10.0, 0, 8))
            cold = 0.0 if (ind or np.isnan(r.temp_f)) else float(np.clip(40.0 - r.temp_f, 0, 30))
            wet = 0.0 if (ind or np.isnan(r.precip_in)) else float(r.precip_in > 0.02)
            f.update(wind=wind, cold=cold, wet=wet, indoor=ind, game_id=r.game_id, season=int(r.season), week=r.week, date=date, home=r.home, away=r.away,
                     margin=r.home_pts - r.away_pts, total=r.home_pts + r.away_pts, spread_close=r.spread_close, spread_open=r.spread_open,
                     total_close=r.total_close, total_open=r.total_open, n_books=r.n_books, matchup=r.matchup, secb1g=r.secb1g)
            rows.append(f)
        eng.update(day)
    return (pd.DataFrame(rows), eng) if engine else pd.DataFrame(rows)


# ------------------------------------------------------------------ models
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


def walk(d, seasons, alpha, total, cols=None):
    out, first = [], d["season"].min()
    cols = cols or (T_BASE if total else M_BASE)
    for s in seasons:
        tr, te = d[(d["season"] < s) & (d["season"] > first)], d[d["season"] == s]
        if len(tr) < 800 or te.empty:
            continue
        t = te.copy()
        t["p"] = Model(cols, alpha, total).fit(tr).predict(te)
        out.append(t)
    return pd.concat(out) if out else None


def mae(r, total):
    return float((r["p"] - r["total" if total else "margin"]).abs().mean())


def tune(g, val, total):
    cur, t0, best = dict(DEFAULT), time.time(), None
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


def lines_cols(tot):
    return ("total_close", "total_open", "total") if tot else ("spread_close", "spread_open", "margin")


def grade(name, r, tot):
    cl, op, y = lines_cols(tot)
    m = r[r[cl].notna()]
    log(f"\n{name}: {len(m)} games with a closing line. Mean miss: model {mae(m, tot):.3f} | closing line {float((m[cl] - m[y]).abs().mean()):.3f}"
        + (f" | opening line {float((m[op] - m[y]).abs().mean()):.3f}" if m[op].notna().any() else ""))
    e = m["p"] - m[cl]
    log(f"  corr of (model - closing line) with the line's miss: {np.corrcoef(e, m[y] - m[cl])[0, 1]:+.3f}   (0 = nothing beyond the line)")
    log("  win rate betting the model's side against the CLOSING line (need > 52.4%):")
    log(win_record(e, m[y] - m[cl], TOTAL_EDGES if tot else SPREAD_EDGES).round(3).to_string(index=False))
    mo = m[m[op].notna()]
    if len(mo) > 100:
        log(f"  against the OPENING line ({len(mo)} games):")
        log(win_record(mo["p"] - mo[op], mo[y] - mo[op], TOTAL_EDGES if tot else SPREAD_EDGES).round(3).to_string(index=False))


def variants_report(d, ev, cur, p, tot, variants, alpha):
    """Each variant, out of sample in every season: the gap to the closing line per season, and the win rate at 3+ points of edge."""
    cl, op, y = lines_cols(tot)
    log(f"\n{'TOTAL' if tot else 'SPREAD'} MODEL: WHAT DOES EACH ADDITION DO?  (gap = model miss minus closing-line miss, in points; smaller is better)")
    seasons = sorted(set(ev) | ({cur} if cur else set()))
    log(f"  {'variant':26s}" + "".join(f"{s:>8d}" for s in seasons) + "     all   win%@2  win%@3  win%@4   (all seasons pooled, n at 3+)")
    out = {}
    for name, cols in variants.items():
        r = walk(d, seasons, alpha, tot, cols)
        r = r[r[cl].notna()]
        out[name] = r
        gaps = {s: mae(x, tot) - float((x[cl] - x[y]).abs().mean()) for s, x in r.groupby("season")}
        rp = r[r["season"].isin(ev)]
        e = rp["p"] - rp[cl]
        wr = win_record(e, rp[y] - rp[cl], [2.0, 3.0, 4.0])
        gap_all = mae(rp, tot) - float((rp[cl] - rp[y]).abs().mean())
        log(f"  {name:26s}" + "".join(f"{gaps.get(s, np.nan):8.2f}" for s in seasons) + f"  {gap_all:6.2f}   "
            + "  ".join(f"{v:5.1%} " for v in wr["win_pct"]) + f"  n={int(wr['games'][1])}")
    return out


def tier_report(r, tot, label):
    cl, op, y = lines_cols(tot)
    r = r[r[cl].notna()].copy()
    log(f"\n{label}: WHERE IS THE MODEL CLOSEST TO THE MARKET?  (pooled out-of-sample seasons {sorted(int(s) for s in r['season'].unique())})")
    thr = [2.0, 3.0, 4.0] if tot else [1.0, 2.0, 3.0]
    log(f"  {'group':28s} {'games':>6s} {'books':>6s} {'mkt miss':>9s} {'model':>7s} {'gap':>6s} {'corr':>6s}   win% vs close at {thr[0]:g}+/{thr[1]:g}+/{thr[2]:g}+   win% vs open at {thr[1]:g}+ (n)")
    for col in ("matchup", "secb1g"):
        for k, x in r.groupby(col):
            if len(x) < 150:
                continue
            e = x["p"] - x[cl]
            wr = win_record(e, x[y] - x[cl], thr)
            xo = x[x[op].notna()]
            wo = win_record(xo["p"] - xo[op], xo[y] - xo[op], [thr[1]]) if len(xo) > 100 else None
            mm = float((x[cl] - x[y]).abs().mean())
            log(f"  {k:28s} {len(x):6d} {x['n_books'].mean():6.1f} {mm:9.3f} {mae(x, tot):7.3f} {mae(x, tot) - mm:6.2f} {np.corrcoef(e, x[y] - x[cl])[0, 1]:+6.2f}   "
                + " / ".join(f"{v:5.1%}" for v in wr["win_pct"]) + (f"   {wo['win_pct'][0]:5.1%} ({int(wo['games'][0])})" if wo is not None else ""))
    # stability: win rate at the middle threshold by season for each matchup type
    log(f"  win rate at {thr[1]:g}+ by season (games):")
    seasons = sorted(r["season"].unique())
    log(f"  {'group':28s}" + "".join(f"{int(s):>12d}" for s in seasons))
    for k, x in r.groupby("matchup"):
        cells = []
        for s in seasons:
            z = x[x["season"] == s]
            w = win_record(z["p"] - z[cl], z[y] - z[cl], [thr[1]])
            cells.append(f"{w['win_pct'][0]:6.1%}({int(w['games'][0]):3d})" if w["games"][0] >= 20 else "      -     ")
        log(f"  {k:28s}" + "".join(f"{c:>12s}" for c in cells))


def main():
    if not os.path.exists(f"{DATA}/cfb_games.csv"):
        log(f"No {DATA}/cfb_games.csv. Run cfb_fetch.py first.")
        return 1
    g = load(DATA)
    played = g[g["played"]]
    have = lambda c: int(played[c].notna().sum())
    log(f"{len(played)} played FBS-vs-FBS games, seasons {int(played['season'].min())}-{int(played['season'].max())}; {have('spread_close')} with a closing spread, "
        f"{have('total_close')} with a closing total, {have('h_plays')} with pace and efficiency stats, {have('wind_mph') + int((played['indoor'] == 1).sum())} with weather")
    ok = played.dropna(subset=["spread_close"])
    sign = 1.0 if np.corrcoef(-ok["spread_close"], ok["home_pts"] - ok["away_pts"])[0, 1] > 0 else -1.0
    log(f"market spread sign check: home spread is {'negative when the home team is favored (as expected)' if sign > 0 else 'POSITIVE when the home team is favored; flipped'}")
    for c in ("spread_close", "spread_open"):
        g[c] = g[c] * -sign
    log("games by group: " + ", ".join(f"{k} {v}" for k, v in played["matchup"].value_counts().items()) + " | "
        + ", ".join(f"{k} {v}" for k, v in played["secb1g"].value_counts().items()))
    d0 = history(g, DEFAULT)
    full = [int(s) for s in sorted(d0["season"].unique()) if (d0["season"] == s).sum() >= 600 and d0.loc[d0["season"] == s, "spread_close"].notna().mean() > 0.6]
    test = full[-N_TEST:]
    val = [s for s in full if s < test[0] and s >= int(d0["season"].min()) + 2]
    ev = [s for s in full if s >= int(d0["season"].min()) + 2]
    cur = int(d0["season"].max())
    cur = cur if (cur not in full and (d0["season"] == cur).sum() >= 150) else None
    log(f"tuning on seasons {val}; grading on {test}; season-by-season tests on {ev}" + (f" plus {cur} so far" if cur else ""))
    if not val or not test:
        log("Not enough seasons.")
        return 1

    log("\nSTEP 1: TUNING the rating settings (mean miss on the actual result over the tuning seasons, lower is better)")
    p_m, s_m = tune(g, val, False)
    p_t, s_t = tune(g, val, True)
    log(f"  margin settings: {p_m} -> {s_m:.3f}\n  total settings:  {p_t} -> {s_t:.3f}")
    dm, dt = history(g, p_m), history(g, p_t)

    log("\nSTEP 2: GRADE the plain points-ratings model on the last seasons")
    grade("SPREAD", walk(dm, test, p_m["ALPHA"], False), False)
    grade("TOTAL", walk(dt, test, p_t["ALPHA"], True), True)

    log("\nSTEP 3: EXTRA INFORMATION, season by season (every season is predicted using only earlier seasons)")
    vm = variants_report(dm, ev, cur, p_m, False, MARGIN_VARIANTS, p_m["ALPHA"])
    vt = variants_report(dt, ev, cur, p_t, True, TOTAL_VARIANTS, p_t["ALPHA"])

    log("\nSTEP 4: SMALLER SCHOOLS vs SEC / BIG TEN, using the model with all the extra information")
    rt = vt["+ all four"]
    rm = vm["+ efficiency ratings"]
    tier_report(rt[rt["season"].isin(ev)], True, "TOTALS")
    tier_report(rm[rm["season"].isin(ev)], False, "SPREADS")

    tr = dt[dt["season"] > dt["season"].min()]
    wts = Model(TOTAL_VARIANTS["+ all four"], p_t["ALPHA"], True).fit(tr).coef
    log("\nTOTAL MODEL WEIGHTS with everything (points per 1 standard deviation of the feature):")
    log(wts.round(2).to_string())
    os.makedirs(BOARD, exist_ok=True)
    keep = ["game_id", "season", "week", "date", "home", "away", "matchup", "secb1g", "n_books", "margin", "total", "spread_close", "spread_open", "total_close", "total_open"]
    a = rm[keep + ["p"]].rename(columns={"p": "p_margin_eff"})
    b = rt[["game_id", "p"]].rename(columns={"p": "p_total_all"})
    a.merge(b, on="game_id", how="outer").to_csv(f"{BOARD}/cfb_backtest_latest.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
