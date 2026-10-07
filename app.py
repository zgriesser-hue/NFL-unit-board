"""Phone-friendly viewer for the weekly unit board. Reads the CSVs the workflow commits to this repo.

Run locally:  streamlit run app.py
Hosted:       Streamlit Community Cloud (free, works with private repos), main file = app.py
Sign convention everywhere: positive = HOME team favored, in points.
"""
import glob
import os

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

tab_week, tab_game, tab_teams, tab_qb, tab_sch, tab_fwd = st.tabs(
    ["This week", "Game", "Teams", "QBs", "Schemes", "Forward test"])

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
        "Model": b["fair_home_spread"].map(fmt_line),
    })
    if has_mkt:
        view["Market"] = b["market_home_spread"].map(fmt_line)
        view["Edge"] = b["edge_vs_market"].round(1)
        view["Model side"] = b.apply(pick_text, axis=1)
        if "fair_at_market_weights" in b:
            view["At mkt wts"] = b["fair_at_market_weights"].map(fmt_line)
            view["Not in data"] = b["market_minus_mkt_weighted"].round(1)
    st.caption("'At mkt wts' is our same data weighted the way the market weights it. 'Not in data' is "
               "market minus that, in points toward the home team: what our data can't explain "
               "(injury news, roster moves, anything not in the stats).")
    st.caption("Lines are the home team's spread (negative = home favored). "
               "Edge = model minus market in points; positive means the model likes the home team more.")
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
    c1.metric("Model", fmt_line(r["fair_home_spread"]))
    if "market_home_spread" in r and pd.notna(r["market_home_spread"]):
        c2.metric("Market", fmt_line(r["market_home_spread"]))
        c3.metric("Edge", f"{r['edge_vs_market']:+.1f}")
        if "fair_at_market_weights" in r and pd.notna(r["fair_at_market_weights"]):
            d1, d2 = st.columns(2)
            d1.metric("Our data, market weights", fmt_line(r["fair_at_market_weights"]))
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
        lg = {lab: round(float(prof[k + "_league"].iloc[0]) * 100) for k, lab in SCH.items()}
        st.dataframe(show.sort_values("Man coverage %", ascending=False), use_container_width=True)
        st.caption("League average: " + ", ".join(f"{lab} {v}%" for lab, v in lg.items()))

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
