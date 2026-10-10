"""Phone-friendly viewer for the weekly unit board. Reads the CSVs the workflow commits to this repo.

Run locally:  streamlit run app.py
Hosted:       Streamlit Community Cloud (free, works with private repos), main file = app.py
Sign convention everywhere: positive = HOME team favored, in points.
"""
import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
UNITS = ["QB", "O-line", "Skill", "Front", "Coverage", "context"]

st.set_page_config(page_title="NFL Unit Board", layout="centered")


def fmt_line(x):
    """Home spread in betting style: favored home = negative number, e.g. +3.5 -> -3.5."""
    if pd.isna(x):
        return "-"
    return "PK" if abs(x) < 0.05 else f"{-x:+.1f}"


def tl(home, away, hs):
    """Spread with the team named: the favorite and its points, e.g. 'WAS -4.5'. Positive hs = home favored."""
    if pd.isna(hs):
        return "-"
    if abs(hs) < 0.05:
        return "PK"
    return f"{home} -{hs:.1f}" if hs > 0 else f"{away} -{-hs:.1f}"


def row_tl(df, col):
    return df.apply(lambda r: tl(r["home_team"], r["away_team"], r[col]), axis=1)


def pick_text(row):
    """Which side the model prefers vs the market, in words."""
    e = row.get("edge_vs_market")
    if pd.isna(e) or abs(e) < 0.05:
        return ""
    team = row["home_team"] if e > 0 else row["away_team"]
    line = row["market_home_spread"] if e > 0 else -row["market_home_spread"]
    return f"{team} {-line:+.1f}"


@st.cache_data(ttl=300)
def load_boards():
    files = sorted(glob.glob(f"{ROOT}/boards/unit_board_20*_wk*_*.csv"))
    out = {}
    for f in files:
        d = pd.read_csv(f)
        out[os.path.basename(f).replace("unit_board_", "").replace(".csv", "")] = d
    return out


@st.cache_data(ttl=300)
def load_csv(name, index_col=None):
    p = f"{ROOT}/{name}"
    return pd.read_csv(p, index_col=index_col) if os.path.exists(p) else None


boards = load_boards()
st.title("NFL Unit Board")

if not boards:
    st.info("No boards yet. Run the workflow from the Actions tab.")
    st.stop()

tab_week, tab_game, tab_teams, tab_qb, tab_sch, tab_wi, tab_fwd, tab_bets = st.tabs(
    ["This week", "Game", "Teams", "QBs", "Schemes", "What-if", "Forward test", "Bets"])

# ---------------------------------------------------------------- This week
with tab_week:
    keys = list(boards)[::-1]
    key = st.selectbox("Board version", keys, index=0,
                       help="Newest first. Each file is one run (Wednesday / Friday).")
    b = boards[key].copy()
    built = pd.to_datetime(b["built_at_utc"].iloc[0], utc=True).tz_convert("America/New_York")
    st.caption(f"Built {built:%a %b %d, %I:%M %p} ET")
    has_mkt = "market_home_spread" in b and b["market_home_spread"].notna().any()
    if not has_mkt:
        st.warning("No market lines on this board.")

    b["kickoff"] = pd.to_datetime(b["kickoff_utc"], utc=True).dt.tz_convert("America/New_York")
    b = b.sort_values("kickoff")
    view = pd.DataFrame({
        "Game": b["away_team"] + " @ " + b["home_team"],
        "Kick": b["kickoff"].dt.strftime("%a %-I:%M"),
        "Model": row_tl(b, "fair_home_spread"),
    })
    if has_mkt:
        view["Market"] = row_tl(b, "market_home_spread")
        view["Edge"] = b["edge_vs_market"].round(1)
        view["Model side"] = b.apply(pick_text, axis=1)
        if "fair_at_market_weights" in b:
            view["At mkt wts"] = row_tl(b, "fair_at_market_weights")
            view["Not in data"] = b["market_minus_mkt_weighted"].round(1)
    st.caption("'At mkt wts' is our same data weighted the way the market weights it. 'Not in data' is "
               "market minus that, in points toward the home team: what our data can't explain "
               "(injury news, roster moves, anything not in the stats).")
    st.caption("Each line names the favorite and how many points it is favored by (WAS -4.5 = Washington favored by 4.5). "
               "Edge = model minus market in points toward the home team; 'Model side' says which team the model likes "
               "more than the market does.")
    if has_mkt:
        view = view.reindex(view["Edge"].abs().sort_values(ascending=False, na_position="last").index)
    st.dataframe(view, hide_index=True, use_container_width=True)

    if has_mkt:
        st.caption("Backtests found no profitable edge from disagreeing with the market, so read "
                   "big edges as 'what the model sees differently' rather than as bets.")

# ---------------------------------------------------------------- Game detail
with tab_game:
    b = boards[list(boards)[-1]].copy()
    labels = (b["away_team"] + " @ " + b["home_team"]).tolist()
    sel = st.selectbox("Game", labels)
    r = b.iloc[labels.index(sel)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", tl(r["home_team"], r["away_team"], r["fair_home_spread"]))
    if "market_home_spread" in r and pd.notna(r["market_home_spread"]):
        c2.metric("Market", tl(r["home_team"], r["away_team"], r["market_home_spread"]))
        c3.metric("Edge", f"{r['edge_vs_market']:+.1f}")
        if "fair_at_market_weights" in r and pd.notna(r["fair_at_market_weights"]):
            d1, d2 = st.columns(2)
            d1.metric("Our data, market weights", tl(r["home_team"], r["away_team"], r["fair_at_market_weights"]))
            d2.metric("Market beyond our data", f"{r['market_minus_mkt_weighted']:+.1f}")
    st.caption(f"QBs: {r['away_QB']} (away) vs {r['home_QB']} (home)")

    contrib = pd.DataFrame({"points toward HOME": [r[u] for u in UNITS if u in r]},
                           index=[u for u in UNITS if u in r])
    st.bar_chart(contrib, horizontal=True)
    st.caption("Each bar is how many points that unit moves the line toward the home team (right) "
               "or the away team (left). The bars sum to the model line.")

    hist = []
    for k, d in boards.items():
        m = d[d["game_id"] == r["game_id"]]
        if len(m):
            hist.append({"Board": k, "Model": m["fair_home_spread"].iloc[0],
                         "Market": m["market_home_spread"].iloc[0] if "market_home_spread" in m else None})
    if len(hist) > 1:
        st.subheader("How the line moved across boards")
        st.line_chart(pd.DataFrame(hist).set_index("Board"))

# ---------------------------------------------------------------- Teams
with tab_teams:
    t = load_csv("boards/unit_team_ratings_latest.csv", index_col=0)
    if t is None:
        st.info("No team ratings yet.")
    else:
        st.caption("Points vs an average team, by unit, after the latest completed games.")
        st.dataframe(t.round(1), use_container_width=True)

# ---------------------------------------------------------------- QBs
with tab_qb:
    qg = load_csv("boards/unit_qb_grades_latest.csv")
    if qg is None:
        st.info("No QB grades yet. They appear after the next board run.")
    else:
        status = st.multiselect("Show", ["starter", "seasoned backup", "limited sample"],
                                default=["starter", "seasoned backup"])
        v = qg[qg["status"].isin(status)].copy()
        team = st.selectbox("Team", ["All"] + sorted(v["team"].dropna().unique().tolist()))
        if team != "All":
            v = v[v["team"] == team]
        v = v.rename(columns={"name": "QB", "team": "Team", "status": "Status",
                              "pts_vs_avg_starter": "Pts vs avg starter", "dropbacks": "Dropbacks",
                              "confidence": "Confidence"})
        st.caption("Points vs the average current starter, using the model's own QB weight. "
                   "Grades come from EPA per dropback (2019 on), pulled toward average when a QB has few "
                   "dropbacks. Confidence is how much his own record counts. Treat backups with low "
                   "confidence as rough.")
        st.dataframe(v[["QB", "Team", "Status", "Pts vs avg starter", "Dropbacks", "Confidence"]]
                     .round({"Pts vs avg starter": 1, "Confidence": 2}),
                     hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- Schemes
SCH = {"blitz": "Blitz", "man": "Man coverage", "cov1": "Cover-1", "cov2": "Cover-2", "cov3": "Cover-3",
       "cov4": "Cover-4"}
DROPBACKS = 36          # typical dropbacks per team per game, for turning EPA into points


def style_label(row):
    parts = []
    if row["man_n"] > 100:
        parts.append("man-heavy" if row["man_z"] >= 1 else "zone-heavy" if row["man_z"] <= -1 else "")
    parts.append("blitz-heavy" if row["blitz_z"] >= 1 else "rarely blitzes" if row["blitz_z"] <= -1 else "")
    return ", ".join(x for x in parts if x) or "balanced"


with tab_sch:
    sp = load_csv("boards/scheme_team_profiles.csv")
    sq = load_csv("boards/scheme_qb_splits.csv")
    if sp is None or sq is None:
        st.info("No scheme data yet. It builds on the next workflow run (or run scheme_profiles.py).")
    else:
        season = st.selectbox("Season", sorted(sp["season"].unique())[::-1], index=0)
        prof = sp[sp["season"] == season].set_index("team")
        st.caption("Shares are the percent of dropbacks faced. 'League' is the same season's average, because the "
                   "charting labels for man/zone shift between seasons. A team's scheme is a stable trait "
                   "(it correlates 0.7-0.9 between odd and even games), so these profiles are reliable.")

        st.subheader("Defensive scheme by team")
        show = pd.DataFrame({"Dropbacks": prof["dropbacks_faced"]})
        for k, lab in SCH.items():
            show[lab + " %"] = (prof[k] * 100).round(0)
        show["Style"] = prof.apply(style_label, axis=1)
        lg = {lab: float(prof[k + "_league"].iloc[0]) * 100 for k, lab in SCH.items()}
        st.dataframe(show.sort_values("Man coverage %", ascending=False, na_position="last"), use_container_width=True)
        st.caption("League average: " + ", ".join(f"{lab} {v:.0f}%" for lab, v in lg.items() if pd.notna(v)))
        missing = [lab for k, lab in SCH.items() if prof[k].notna().sum() == 0]
        if missing:
            st.info(f"{', '.join(missing)} not available for {season} yet: nflverse publishes the coverage charting "
                    "(man/zone and coverage shells) with a delay. Pick the previous season to see those columns.")

        st.subheader("Quarterback vs a defense's scheme")
        st.warning("Treat this as context, not a prediction. In testing, QB-specific scheme splits did not hold up "
                   "out of sample (split-half correlation near zero, and they failed to replicate across eras). "
                   "The model does not use them.")
        mode = st.radio("Matchup", ["From this week's board", "Pick any QB and defense"], horizontal=True)
        if mode == "From this week's board":
            b = boards[list(boards)[-1]]
            pairs = []
            for _, r in b.iterrows():
                pairs += [(r["home_QB"], r["away_team"]), (r["away_QB"], r["home_team"])]
            lbl = [f"{q} vs {t} defense" for q, t in pairs]
            pick = st.selectbox("Matchup", lbl)
            qb_name, dteam = pairs[lbl.index(pick)]
        else:
            c1, c2 = st.columns(2)
            qb_name = c1.selectbox("Quarterback", sorted(sq["name"].dropna().unique()))
            dteam = c2.selectbox("Defense", sorted(prof.index))
        qrows = sq[sq["name"] == qb_name]
        if dteam not in prof.index:
            st.info(f"No scheme profile for {dteam}.")
        elif qrows.empty:
            st.info(f"No scheme splits for {qb_name} (needs about 100 recent charted dropbacks).")
        else:
            st.caption(f"{qb_name}: EPA per dropback vs an average defense {float(qrows['baseline'].iloc[0]):+.2f}. "
                       f"Defense: {dteam}, {season} profile.")
            out = []
            for k, lab in SCH.items():
                q = qrows[qrows["scheme"] == k]
                if q.empty or pd.isna(prof.loc[dteam, k]):
                    continue
                q = q.iloc[0]
                dshare, lshare = float(prof.loc[dteam, k]), float(prof.loc[dteam, k + "_league"])
                out.append({
                    "Scheme": lab,
                    "Defense uses": f"{dshare * 100:.0f}% (league {lshare * 100:.0f}%)",
                    "His EPA with / without": f"{q['epa_yes']:+.2f} / {q['epa_no']:+.2f} (n {int(q['n_yes'])}/{int(q['n_no'])})",
                    "His split vs typical": f"{q['rel_diff']:+.2f} +/- {1.96 * q['se']:.2f}",
                    "Est. effect, pts": round(float(q["shrunk"]) * (dshare - lshare) * DROPBACKS, 2),
                    "Raw effect, pts": round(float(q["rel_diff"]) * (dshare - lshare) * DROPBACKS, 2)})
            if out:
                st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
                st.caption("'His split vs typical' is how much worse (negative) or better he does against that scheme "
                           "than the average QB does, in EPA per dropback, with a 95% margin of error. 'Est. effect' "
                           "pulls that split toward zero by how noisy it is, then multiplies by how much more or less "
                           "the defense uses the scheme than the league (about 36 dropbacks per game). The schemes "
                           "overlap (a man defense is also Cover-1), so do not add the rows together. A margin of "
                           "error wider than the split itself means the number is mostly noise.")

# ---------------------------------------------------------------- What-if
WI_FEAT = {"QB": "qb_inj", "OL": "ol_inj", "WR": "sk_inj", "TE": "sk_inj", "RB": "sk_inj", "DL": "fr_inj",
           "LB": "fr_inj", "DB": "cov_inj"}
WI_UNIT = {"qb_inj": "QB", "ol_inj": "O-line", "sk_inj": "Skill", "fr_inj": "Front", "cov_inj": "Coverage"}
WI_W = {"Healthy": 0.0, "Questionable": 0.25, "Doubtful": 0.85, "Out": 1.0}
GRP_ORDER = {g: i for i, g in enumerate(["QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"])}


def wi_label(w):
    return "Out" if w >= 0.95 else "Doubtful" if w >= 0.6 else "Questionable" if w >= 0.2 else "Healthy"


with tab_wi:
    pool = load_csv("boards/injury_player_pool_latest.csv")
    wts = load_csv("boards/model_weights_latest.csv")
    qbg = load_csv("boards/unit_qb_grades_latest.csv")
    if pool is None or wts is None or qbg is None:
        st.info("The What-if tool needs a fresh workflow run (it builds the player list and model weights).")
    else:
        st.caption("Change a player's availability and see how the model line moves, using the same weights and injury "
                   "math as the board. Nothing is saved. To make a change stick in the board itself, put it in "
                   "injury_overrides.csv (and qb_overrides.csv for a different starting QB). 'Snap % (last 3)' is what the line math "
                   "uses; 'Usual role %' is his typical share over his last 6 games, and 'Check' flags starters who look "
                   "absent or hurt but are not on the report (the board treats those as available).")
        b = boards[list(boards)[-1]].copy()
        labels = (b["away_team"] + " @ " + b["home_team"]).tolist()
        sel = st.selectbox("Game ", labels, key="wi_game")
        r = b.iloc[labels.index(sel)]
        coef = dict(zip(wts["feature"], wts["coef"]))
        grades = qbg.drop_duplicates("name").set_index("name")["pts_vs_avg_starter"]
        deltas, notes, qb_pts, picks = {}, [], {}, {}
        for side, team, qbn, src in (("home", r["home_team"], r["home_QB"], r.get("home_QB_src", "")),
                                     ("away", r["away_team"], r["away_QB"], r.get("away_QB_src", ""))):
            with st.expander(f"{team} ({side}) availability", expanded=True):
                opts = [qbn] + [n for n in qbg.sort_values("pts_vs_avg_starter", ascending=False)["name"] if n != qbn]
                qb_pick = st.selectbox(f"{team} starting QB", opts, key=f"wi_qb_{side}_{team}")
                p = pool[pool["team"] == team].copy()
                p["lab"] = p["w_now"].map(wi_label)
                p["ord"] = p["grp"].map(GRP_ORDER)
                p = p.sort_values(["ord", "snap_share"], ascending=[True, False])
                if "role_share" not in p:
                    p["role_share"], p["last_wk"], p["games_missed"] = p["snap_share"], np.nan, 0
                p = p[(p["snap_share"] >= 0.3) | (p["role_share"] >= 0.3)].reset_index(drop=True)
                chk = np.where((p["games_missed"] >= 2) & (p["w_now"] == 0),
                               "Not on report but missed " + p["games_missed"].astype(int).astype(str) + " straight: likely out?",
                               np.where((p["games_missed"] == 1) & (p["w_now"] == 0), "Missed last game", ""))
                chk = np.where((p["role_share"] - p["snap_share"] >= 0.2) & (chk == ""), "Recent games low (hurt?)", chk)
                edit_in = pd.DataFrame({"Player": p["name"], "Pos": p["position"],
                                        "Snap % (last 3)": (p["snap_share"] * 100).round(0),
                                        "Usual role %": (p["role_share"] * 100).round(0),
                                        "Last played": p["last_wk"].map(lambda x: "" if pd.isna(x) else f"Wk {int(x)}"),
                                        "Check": chk, "Now": p["lab"], "Your call": p["lab"]})
                ed = st.data_editor(
                    edit_in, hide_index=True, use_container_width=True, key=f"wi_ed_{side}_{team}",
                    disabled=["Player", "Pos", "Snap % (last 3)", "Usual role %", "Last played", "Check", "Now"],
                    column_config={"Your call": st.column_config.SelectboxColumn(
                        "Your call", options=list(WI_W), required=True)})
                qb_changed = qb_pick != qbn
                qb_override_now = str(src).startswith(("override", "auto"))
                d = {f: 0.0 for f in WI_UNIT}
                for i, row in ed.iterrows():
                    if row["Your call"] == row["Now"]:
                        continue
                    f = WI_FEAT.get(p.loc[i, "grp"])
                    if f is None or (f == "qb_inj" and (qb_changed or qb_override_now)):
                        continue
                    d[f] += (WI_W[row["Your call"]] - float(p.loc[i, "w_now"])) * float(p.loc[i, "snap_share"])
                if qb_changed and not qb_override_now:                 # a new QB replaces the QB injury penalty
                    d["qb_inj"] -= float((p[p["grp"] == "QB"]["w_now"] * p[p["grp"] == "QB"]["snap_share"]).sum())
                deltas[side] = d
                qb_pts[side] = 0.0
                if qb_changed:
                    if qb_pick in grades.index and qbn in grades.index:
                        qb_pts[side] = float(grades[qb_pick] - grades[qbn])
                    else:
                        notes.append(f"No grade found for {qb_pick if qb_pick not in grades.index else qbn}: QB swap ignored for {team}.")
        change = 0.0
        rows = []
        for f, unit in WI_UNIT.items():
            v = coef.get(f, 0.0) * (-deltas["home"][f] + deltas["away"][f])
            change += v
            if abs(v) > 0.005:
                rows.append({"Where": unit + " injuries", "Points toward home": round(v, 2)})
        qv = qb_pts["home"] - qb_pts["away"]
        change += qv
        if abs(qv) > 0.005:
            rows.append({"Where": "Starting QB change", "Points toward home": round(qv, 2)})
        base = float(r["fair_home_spread"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Board line", tl(r["home_team"], r["away_team"], base))
        c2.metric("What-if line", tl(r["home_team"], r["away_team"], base + change), delta=f"{change:+.1f} pts toward home", delta_color="off")
        if "market_home_spread" in r and pd.notna(r["market_home_spread"]):
            c3.metric("Edge vs market", f"{base + change - r['market_home_spread']:+.1f}",
                      delta=f"was {base - r['market_home_spread']:+.1f}", delta_color="off")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        for n in notes:
            st.warning(n)
        st.caption("Positive = more toward the home team. The model weights injuries by status (Out 1.0, Doubtful 0.85, "
                   "Questionable 0.25) times the player's recent snap share, so a player with a low snap share barely "
                   "moves the line. These weights were fit to the final injury report, so a status you set by hand is "
                   "only as good as your read on who will really play.")

# ---------------------------------------------------------------- Forward test
with tab_fwd:
    games = load_csv("forward_test_games.csv")
    summ = load_csv("forward_test_summary.csv")
    if games is None:
        st.info("Nothing graded yet. Grading starts after games finish that have a saved pregame board "
                "(the Tuesday run does it).")
    else:
        games = games.copy()
        c1, c2, c3 = st.columns(3)
        c1.metric("Graded games", len(games))
        c2.metric("Model MAE", f"{(games['fair_home_spread'] - games['margin']).abs().mean():.2f}")
        if "market_home_spread" in games and games["market_home_spread"].notna().any():
            h = games.dropna(subset=["market_home_spread"])
            c3.metric("Market MAE", f"{(h['market_home_spread'] - h['margin']).abs().mean():.2f}")
        st.caption("MAE = average miss in points vs the actual margin; lower is better.")
        if summ is not None and len(summ):
            st.subheader("Betting the model side vs the market")
            s = summ.copy()
            s["win_rate"] = (s["win_rate"] * 100).round(1).astype(str) + "%"
            s["roi_at_110"] = (s["roi_at_110"] * 100).round(1).astype(str) + "%"
            st.dataframe(s, hide_index=True, use_container_width=True)
            st.caption("Break-even at -110 is 52.4%. Small samples swing wildly, so wait for many weeks.")
        cols = [c for c in ["week", "away_team", "home_team", "fair_home_spread", "market_home_spread",
                            "margin"] if c in games]
        st.subheader("Graded games")
        st.dataframe(games[cols].round(1), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- Bet tracker
def bet_profit(stake, odds, outcome):
    if outcome == "Win":
        return stake * (100.0 / abs(odds) if odds < 0 else odds / 100.0)
    return -stake if outcome == "Loss" else 0.0


with tab_bets:
    BET_COLS = ["placed", "game_id", "season", "week", "away_team", "home_team", "pick", "line", "odds", "stake",
                "book", "note", "model_home", "market_home"]
    if "bets" not in st.session_state:
        st.session_state["bets"] = pd.DataFrame(columns=BET_COLS)
    st.caption("Your bets live in this browser session only (nothing is saved to the repo or the server). "
               "Download the file after you add bets, and load it here next time.")
    up = st.file_uploader("Load saved bets (my_bets.csv)", type="csv", key="bets_up")
    if up is not None and st.session_state.get("bets_up_id") != f"{up.name}-{up.size}":
        try:
            loaded = pd.read_csv(up)
            st.session_state["bets"] = loaded.reindex(columns=BET_COLS)
            st.session_state["bets_up_id"] = f"{up.name}-{up.size}"
        except Exception as e:
            st.error(f"Could not read that file: {str(e)[:100]}")

    latest = boards[list(boards)[-1]].copy()
    labels = (latest["away_team"] + " @ " + latest["home_team"]).tolist()
    with st.form("bet_form", clear_on_submit=False):
        st.subheader("Add a bet")
        f1, f2 = st.columns(2)
        game = f1.selectbox("Game", labels)
        gi = labels.index(game)
        away, home = latest.iloc[gi]["away_team"], latest.iloc[gi]["home_team"]
        pick = f2.selectbox("Team you bet", [away, home])
        f3, f4, f5 = st.columns(3)
        line = f3.number_input("Line you got (for your team; -3.5 = laying 3.5, +3.5 = getting 3.5)",
                               value=0.0, step=0.5, format="%.1f")
        odds = f4.number_input("Odds (American)", value=-110, step=5)
        stake = f5.number_input("Stake (units or $)", value=1.0, min_value=0.0, step=0.5)
        f6, f7 = st.columns(2)
        book = f6.text_input("Book (optional)")
        note = f7.text_input("Note (optional)")
        go = st.form_submit_button("Add bet")
    if go:
        row = latest.iloc[gi]
        new = pd.DataFrame([{
            "placed": pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M"),
            "game_id": row["game_id"], "season": int(str(row["game_id"])[:4]), "week": int(str(row["game_id"])[5:7]),
            "away_team": away, "home_team": home, "pick": pick, "line": line, "odds": int(odds), "stake": stake,
            "book": book, "note": note, "model_home": row.get("fair_home_spread"),
            "market_home": row.get("market_home_spread")}])
        st.session_state["bets"] = pd.concat([st.session_state["bets"], new], ignore_index=True)
        st.success(f"Added {pick} {line:+.1f} ({int(odds):+d}), stake {stake:g}")

    bets = st.session_state["bets"].copy()
    if bets.empty:
        st.info("No bets yet. Add one above, or load a saved file.")
    else:
        res = load_csv("boards/results_latest.csv")
        if res is not None:
            bets = bets.merge(res[["game_id", "home_score", "away_score"]], on="game_id", how="left")
        else:
            bets["home_score"], bets["away_score"] = np.nan, np.nan
        for c in ("line", "odds", "stake", "model_home", "market_home"):
            bets[c] = pd.to_numeric(bets[c], errors="coerce")
        is_home = bets["pick"] == bets["home_team"]
        pm = np.where(is_home, bets["home_score"] - bets["away_score"], bets["away_score"] - bets["home_score"])
        cover = pm + bets["line"]                                 # > 0 means the bet covered
        bets["result"] = np.where(bets["home_score"].isna(), "Pending",
                                  np.where(cover > 0, "Win", np.where(cover < 0, "Loss", "Push")))
        bets["profit"] = [bet_profit(s, o, r) if r != "Pending" else np.nan
                          for s, o, r in zip(bets["stake"], bets["odds"], bets["result"])]
        pick_fair = np.where(is_home, bets["model_home"], -bets["model_home"])
        bets["model_view"] = np.where(pd.isna(pick_fair), "", np.where(pick_fair + bets["line"] > 0, "With model", "Against model"))
        done = bets[bets["result"] != "Pending"]
        decided = done[done["result"] != "Push"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Record", f"{(done['result'] == 'Win').sum()}-{(done['result'] == 'Loss').sum()}-{(done['result'] == 'Push').sum()}")
        m2.metric("Profit", f"{done['profit'].sum():+.2f}")
        risked = done["stake"].sum()
        m3.metric("ROI", f"{done['profit'].sum() / risked:+.1%}" if risked else "-")
        m4.metric("Pending", int((bets["result"] == "Pending").sum()))
        if len(decided):
            wr = float((decided["result"] == "Win").mean())
            be = float((decided["odds"].map(lambda o: abs(o) / (abs(o) + 100) if o < 0 else 100 / (o + 100))).mean())
            st.caption(f"Win rate {wr:.1%} on {len(decided)} decided bets; break-even at your average odds is {be:.1%}. "
                       "Under about 100 bets the record says little either way.")
            by = decided.groupby("model_view").agg(bets=("result", "size"), wins=("result", lambda x: (x == "Win").sum()),
                                                   profit=("profit", "sum")).reset_index()
            by = by[by["model_view"] != ""]
            if len(by):
                by["win_rate"] = (by["wins"] / by["bets"] * 100).round(1)
                st.subheader("With vs against the model")
                st.dataframe(by.rename(columns={"model_view": "Your bet was"}).round(2), hide_index=True,
                             use_container_width=True)
            cum = done.reset_index(drop=True)["profit"].cumsum()
            st.subheader("Profit over time")
            st.line_chart(cum, height=200)
        st.subheader("All bets")
        show = bets[["placed", "week", "away_team", "home_team", "pick", "line", "odds", "stake", "result", "profit",
                     "model_view", "book", "note"]].iloc[::-1]
        st.dataframe(show.round(2), hide_index=True, use_container_width=True)
        with st.expander("Fix or delete a bet"):
            raw = st.session_state["bets"].reset_index(drop=True)
            ed = st.data_editor(raw, num_rows="dynamic", hide_index=True, use_container_width=True, key="bets_edit",
                                disabled=["game_id", "season", "week", "away_team", "home_team", "model_home", "market_home"])
            if st.button("Save changes"):
                st.session_state["bets"] = ed
                st.rerun()
        st.download_button("Download my_bets.csv", st.session_state["bets"].to_csv(index=False),
                           file_name="my_bets.csv", mime="text/csv")
