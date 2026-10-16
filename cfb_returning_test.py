"""Two college football tests, out of sample (every season is predicted using only earlier seasons). Writes boards/cfb_returning_test_latest.txt.

1. RETURNING PRODUCTION: does carrying over more of last season's rating for teams that bring back more production (and less for teams
   that lost it) beat the flat carry-over? K = how strongly the carry-over responds (0 = today's model).
2. MARKET BLEND: how much weight does the model earn on top of the closing line? margin = a + b*market + c*(model - market). c = 0 means the
   model adds nothing; c = 1 means trust the model fully. Then the average miss of the blends market + w*(model - market).
Gap = model miss minus closing-line miss in points (smaller is better; 0 = as good as the market).
"""
import os
import sys

import numpy as np
import pandas as pd

import cfb_model_pipeline as P

OUTLINES = []


def say(m=""):
    print(m, flush=True)
    OUTLINES.append(m)


PM = {"K_PRIOR": 4.0, "DECAY": 1.0, "CARRY": 0.9, "ALPHA": 30.0}          # same fixed settings as the college board
PT = {"K_PRIOR": 14.0, "DECAY": 1.0, "CARRY": 0.6, "ALPHA": 100.0}
RET = {}


class ReturningRatings(P.Ratings):
    """Same ratings, but the carry-over into a new season moves with how much of last season's production the team returns."""
    K = 0.0

    def new_season(self, season):
        if self.season is not None:
            c = self.p["CARRY"]
            vals = [RET[(season, tm)] for tm in self.t if (season, tm) in RET]
            avg = float(np.mean(vals)) if vals else 0.0
            for tm in list(self.t):
                r = RET.get((season, tm), avg)
                ci = float(np.clip(c + self.K * (r - avg), 0.0, 1.0))
                self.prior[tm] = tuple(ci * x for x in self.get(tm))
            for m in self.lg.values():
                m.new_season()
        self.t, self.season = {}, season


def run_variant(g, p, total, cols, seasons, k):
    ReturningRatings.K = k
    P.Ratings = ReturningRatings
    d = P.history(g, p)
    return P.walk(d, seasons, p["ALPHA"], total, cols)


def summarize(r, total, label, seasons):
    cl, op, y = P.lines_cols(total)
    r = r[r[cl].notna()]
    gaps = {s: P.mae(x, total) - float((x[cl] - x[y]).abs().mean()) for s, x in r.groupby("season")}
    allg = P.mae(r, total) - float((r[cl] - r[y]).abs().mean())
    thr = 3.0 if total else 2.0
    wr = P.win_record(r["p"] - r[cl], r[y] - r[cl], [thr])
    say(f"  {label:34s}" + "".join(f"{gaps.get(s, np.nan):8.2f}" for s in seasons) + f"  {allg:7.2f}   {wr['win_pct'][0]:5.1%} (n={int(wr['games'][0])})")
    return allg


def blend_report(r, total, name):
    cl, op, y = P.lines_cols(total)
    r = r[r[cl].notna()].copy()
    mk = r[cl] if total else r[cl]                      # spread_close is already flipped so + = home favored = predicted margin
    e = r["p"] - mk
    X = np.column_stack([np.ones(len(r)), mk.values, e.values])
    beta, *_ = np.linalg.lstsq(X, r[y].values, rcond=None)
    res = r[y].values - X @ beta
    cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(r) - 3)
    se = np.sqrt(np.diag(cov))
    say(f"\n{name}: actual = a + b*market + c*(model - market), {len(r)} games, seasons {sorted(int(s) for s in r['season'].unique())}")
    say(f"  b (weight on the market) = {beta[1]:.2f} +/- {1.96 * se[1]:.2f};  c (weight the model earns on top) = {beta[2]:+.2f} +/- {1.96 * se[2]:.2f}")
    for s, x in r.groupby("season"):
        xe = x["p"] - x[cl]
        Xs = np.column_stack([np.ones(len(x)), x[cl].values, xe.values])
        bs, *_ = np.linalg.lstsq(Xs, x[y].values, rcond=None)
        say(f"    {int(s)}: c = {bs[2]:+.2f} ({len(x)} games)")
    say("  average miss of market + w*(model - market):  " + ", ".join(
        f"w={w:g}: {float((mk + w * e - r[y]).abs().mean()):.3f}" for w in (0, 0.25, 0.5, 0.75, 1.0)) + f"   (closing line alone = w=0)")


def main():
    global RET
    g = P.load(P.DATA)
    played = g[g["played"]]
    sign = P.market_sign(played)
    for c in ("spread_close", "spread_open"):
        g[c] = g[c] * -sign
    rp = f"{P.DATA}/cfb_returning.csv"
    if os.path.exists(rp):
        rr = pd.read_csv(rp)
        rr["pct_ppa"] = pd.to_numeric(rr["pct_ppa"], errors="coerce")
        RET = {(int(a), b): float(v) for a, b, v in zip(rr["season"], rr["team"], rr["pct_ppa"]) if pd.notna(v)}
        say(f"returning production loaded for {len(RET)} team-seasons")
    else:
        say("(no data/cfb_returning.csv: skipping the returning-production test)")
    d0 = P.history(g, PM)
    first = int(d0["season"].min())
    ev = [int(s) for s in sorted(d0["season"].unique()) if (d0["season"] == s).sum() >= 600 and d0.loc[d0["season"] == s, "spread_close"].notna().mean() > 0.6 and s >= first + 2]
    cur = int(d0["season"].max())
    seasons = ev + ([cur] if cur not in ev and (d0["season"] == cur).sum() >= 150 else [])
    cols_m, cols_t = P.M_BASE + P.EFF, P.T_BASE + P.PACE + P.EFF
    base = {}
    if RET:
        say("\nRETURNING PRODUCTION: gap to the closing line by season (model miss minus market miss, points; smaller is better)")
        say(f"  {'variant':34s}" + "".join(f"{s:>8d}" for s in seasons) + "      all   win% at 2+ (spread) / 3+ (total)")
        for tot, p, cols, nm in ((False, PM, cols_m, "SPREAD"), (True, PT, cols_t, "TOTAL")):
            say(f"  {nm}")
            for k in (0.0, 0.5, 1.0, 2.0):
                r = run_variant(g, p, tot, cols, seasons, k)
                summarize(r, tot, "today's carry-over (K=0)" if k == 0 else f"carry-over moves with returning, K={k:g}", seasons)
                if k == 0:
                    base[tot] = r
    else:
        ReturningRatings.K = 0.0
        P.Ratings = ReturningRatings
        base[False] = P.walk(P.history(g, PM), seasons, PM["ALPHA"], False, cols_m)
        base[True] = P.walk(P.history(g, PT), seasons, PT["ALPHA"], True, cols_t)
    blend_report(base[False], False, "\nMARKET BLEND, SPREADS")
    blend_report(base[True], True, "MARKET BLEND, TOTALS")
    os.makedirs(P.BOARD, exist_ok=True)
    with open(f"{P.BOARD}/cfb_returning_test_latest.txt", "w") as f:
        f.write("\n".join(OUTLINES) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
