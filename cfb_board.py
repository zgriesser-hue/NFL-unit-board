"""College football projections board: grades this season so far and projects the upcoming games that already have a market line.

Uses the settings found by cfb_model_pipeline.py. Every prediction for a game uses only games played before it (and, for the
regressions, only earlier seasons). Weather is not used for upcoming games (no forecast), so the totals board uses pace + efficiency.
Writes boards/cfb_board_latest.csv (upcoming) and boards/cfb_board_graded_latest.csv (this season's games, graded).
"""
import os
import sys

import numpy as np
import pandas as pd

import cfb_model_pipeline as P

PM = {"K_PRIOR": 4.0, "DECAY": 1.0, "CARRY": 0.9, "ALPHA": 30.0}
PT = {"K_PRIOR": 14.0, "DECAY": 1.0, "CARRY": 0.6, "ALPHA": 100.0}
DAYS = int(os.environ.get("CFB_BOARD_DAYS", "9"))
log = P.log


def record(df, edge, res, thr):
    rows = []
    for t in thr:
        m = edge.abs() >= t
        r = res[m]
        r = r[r != 0]
        w = int((np.sign(edge[m][res[m] != 0]) * r > 0).sum())
        rows.append(f"  edge >= {t:g}: {w}-{len(r) - w}" + (f" ({w / len(r):.1%})" if len(r) else ""))
    return "\n".join(rows)


def main():
    g = P.load(P.DATA)
    played = g[g["played"]]
    ok = played.dropna(subset=["spread_close"])
    sign = P.market_sign(played)
    for c in ("spread_close", "spread_open"):
        g[c] = g[c] * -sign          # now: positive = home favored by that many points
    dm, em = P.history(g, PM, engine=True)
    dt, et = P.history(g, PT, engine=True)
    cur = int(dm["season"].max())
    cols_m, cols_t = P.M_BASE + P.EFF, P.T_BASE + P.PACE + P.EFF
    # ---- grade this season (walk-forward: trained on earlier seasons only)
    gm = P.walk(dm, [cur], PM["ALPHA"], False, cols_m)
    gt = P.walk(dt, [cur], PT["ALPHA"], True, cols_t)
    gr = gm[["game_id", "week", "date", "home", "away", "margin", "spread_close", "spread_open", "p"]].rename(columns={"p": "proj_margin"}).merge(
        gt[["game_id", "total", "total_close", "total_open", "p"]].rename(columns={"p": "proj_total"}), on="game_id", how="left")
    os.makedirs(P.BOARD, exist_ok=True)
    gr.to_csv(f"{P.BOARD}/cfb_board_graded_latest.csv", index=False)
    log(f"THIS SEASON ({cur}) SO FAR: {len(gr)} games, weeks {int(gr['week'].min())}-{int(gr['week'].max())}; each projected with only earlier information")
    for tag, col in (("closing", "close"), ("opening", "open")):
        d = gr.dropna(subset=[f"spread_{col}"])
        log(f"\nSPREAD vs {tag} line: {len(d)} games | average miss: model {(d['margin'] - d['proj_margin']).abs().mean():.2f}, line {(d['margin'] - d[f'spread_{col}']).abs().mean():.2f}")
        log(record(d, d["proj_margin"] - d[f"spread_{col}"], d["margin"] - d[f"spread_{col}"], [1, 2, 3, 4]))
    for tag, col in (("closing", "close"), ("opening", "open")):
        d = gr.dropna(subset=[f"total_{col}"])
        log(f"\nTOTAL vs {tag} line: {len(d)} games | average miss: model {(d['total'] - d['proj_total']).abs().mean():.2f}, line {(d['total'] - d[f'total_{col}']).abs().mean():.2f}")
        log(record(d, d["proj_total"] - d[f"total_{col}"], d["total"] - d[f"total_{col}"], [2, 3, 4, 6]))
    by = gr.dropna(subset=["spread_close"]).assign(e=lambda d: d["proj_margin"] - d["spread_close"], r=lambda d: d["margin"] - d["spread_close"])
    by = by[(by["e"].abs() >= 2) & (by["r"] != 0)]
    log("\nSPREAD picks (edge 2+ vs close) by week: " + ", ".join(f"wk{int(w)} {int((np.sign(x['e']) * x['r'] > 0).sum())}-{int((np.sign(x['e']) * x['r'] < 0).sum())}" for w, x in by.groupby("week")))

    # ---- board for upcoming games that already have a line
    today = pd.Timestamp.now().normalize()
    up = g[(~g["played"]) & (g["season"] == cur) & (g["date"] >= today) & (g["date"] <= today + pd.Timedelta(days=DAYS))
           & (g["spread_close"].notna() | g["total_close"].notna())]
    if up.empty:
        log("\nNo upcoming games with lines in the next days.")
        return 0
    trm = dm[(dm["season"] < cur) & (dm["season"] > dm["season"].min())]
    trt = dt[(dt["season"] < cur) & (dt["season"] > dt["season"].min())]
    mm, mt = P.Model(cols_m, PM["ALPHA"], False).fit(trm), P.Model(cols_t, PT["ALPHA"], True).fit(trt)
    def note(tm):
        t = em._team(tm)
        cur, pri = em.get(tm), t["p"]
        shift = (cur[0] - cur[1]) - (pri[0] - pri[1])     # points per game: this season's rating minus last season's carried rating
        bits = []
        if t["w"] < 4:
            bits.append(f"{tm}: only {t['w']:.0f} games")
        if abs(shift) >= 6:
            bits.append(f"{tm}: rating {'up' if shift > 0 else 'down'} {abs(shift):.0f} vs last year")
        return "; ".join(bits)

    rows = []
    for r in up.itertuples():
        fm, ft = em.features(r.home, r.away, r.date, r.neutral), et.features(r.home, r.away, r.date, r.neutral)
        pm = float(mm.predict(pd.DataFrame([fm]))[0])
        pt = float(mt.predict(pd.DataFrame([ft]))[0])
        sp, tl = r.spread_close, r.total_close
        rows.append({"week": r.week, "date": r.date.date(), "matchup": f"{r.away} at {r.home}", "type": r.matchup,
                     "line_home_by": sp, "open_home_by": r.spread_open, "open_total": r.total_open, "proj_home_by": round(pm, 1),
                     "spread_edge": round(pm - sp, 1) if pd.notna(sp) else np.nan,
                     "spread_pick": ("" if pd.isna(sp) else (f"{r.home}" if pm > sp else f"{r.away}")),
                     "line_total": tl, "proj_total": round(pt, 1), "total_edge": round(pt - tl, 1) if pd.notna(tl) else np.nan,
                     "caution": "; ".join(x for x in (note(r.home), note(r.away)) if x),
                     "total_pick": ("" if pd.isna(tl) else ("Over" if pt > tl else "Under"))})
    b = pd.DataFrame(rows).sort_values(["date", "matchup"])
    b.to_csv(f"{P.BOARD}/cfb_board_latest.csv", index=False)
    log(f"\nBOARD: {len(b)} upcoming games with lines (lines are the latest available, sign: positive = home favored)")
    show = b.assign(se=b["spread_edge"].abs(), te=b["total_edge"].abs())
    log("\nBiggest TOTAL edges:\n" + show.sort_values("te", ascending=False).head(8)[["date", "matchup", "line_total", "proj_total", "total_pick", "caution"]].to_string(index=False))
    log("\nBiggest SPREAD edges:\n" + show.sort_values("se", ascending=False).head(8)[["date", "matchup", "line_home_by", "proj_home_by", "spread_pick", "caution"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
