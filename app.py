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

tab_week, tab_game, tab_teams, tab_fwd = st.tabs(["This week", "Game", "Teams", "Forward test"])

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
