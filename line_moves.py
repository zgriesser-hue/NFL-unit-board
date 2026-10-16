"""How far has each NFL line moved since the first board that had it? A big move is usually news (an injury, a quarterback change, sharp money)
that the model may not know about. Writes boards/line_moves_latest.csv: game_id, first-seen time, spread_first / spread_now / spread_move
(positive = moved toward the home team), total_first / total_now / total_move (positive = total went up).
Uses the dated board files already in boards/, so the first line is the one the first board of that week saw (Monday for totals)."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("NFL_ROOT", os.getcwd())
B = f"{ROOT}/boards"


def first_lines(latest, prefix, line_col, id_col="game_id"):
    """For each game in `latest`, the earliest dated board (same season/week) with a non-empty line."""
    out = {}
    for gid in latest[id_col]:
        sn, wk = gid.split("_")[0], int(gid.split("_")[1])
        out.setdefault((sn, wk), []).append(gid)
    res = {}
    for (sn, wk), gids in out.items():
        files = sorted(glob.glob(f"{B}/{prefix}_{sn}_wk{wk}_*.csv"))
        for f in files:                                              # oldest first: the first file holding a line for the game wins
            d = pd.read_csv(f)
            if line_col not in d:
                continue
            ts = pd.to_datetime(d["market_captured_utc"] if "market_captured_utc" in d else d.get("built_at_utc"), utc=True, errors="coerce")
            for gid, v, t in zip(d[id_col], d[line_col], ts):
                if gid in gids and gid not in res and pd.notna(v):
                    res[gid] = (float(v), t)
    return res


def main():
    ub = pd.read_csv(f"{B}/unit_board_latest.csv") if os.path.exists(f"{B}/unit_board_latest.csv") else None
    tb = pd.read_csv(f"{B}/totals_board_latest.csv") if os.path.exists(f"{B}/totals_board_latest.csv") else None
    rows = {}
    if ub is not None and "market_home_spread" in ub:
        fs = first_lines(ub, "unit_board", "market_home_spread")
        for gid, now in zip(ub["game_id"], ub["market_home_spread"]):
            r = rows.setdefault(gid, {"game_id": gid})
            if gid in fs and pd.notna(now):
                r.update(spread_first=fs[gid][0], spread_now=float(now), spread_move=float(now) - fs[gid][0], spread_first_seen=fs[gid][1])
    if tb is not None and "market_total" in tb:
        ft = first_lines(tb, "totals_board", "market_total")
        for gid, now in zip(tb["game_id"], tb["market_total"]):
            r = rows.setdefault(gid, {"game_id": gid})
            if gid in ft and pd.notna(now):
                r.update(total_first=ft[gid][0], total_now=float(now), total_move=float(now) - ft[gid][0], total_first_seen=ft[gid][1])
    out = pd.DataFrame(list(rows.values()))
    if out.empty:
        print("no lines to compare")
        return 0
    os.makedirs(B, exist_ok=True)
    out.to_csv(f"{B}/line_moves_latest.csv", index=False)
    big = out[(out.get("spread_move", pd.Series(dtype=float)).abs() >= 1.5) | (out.get("total_move", pd.Series(dtype=float)).abs() >= 2)]
    print(f"line moves for {len(out)} games; {len(big)} moved a lot:")
    for _, r in big.iterrows():
        print("  ", r["game_id"], f"spread {r.get('spread_move', np.nan):+.1f}", f"total {r.get('total_move', np.nan):+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
