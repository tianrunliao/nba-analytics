"""Live page: the remaining-season projection from data fetched when the page is opened.

    uv run streamlit run app/streamlit_app.py

Opening or refreshing the page runs the live flow (nba_sos.live.run_live: snapshot, latest
official injury report, projection) once for that browser session; "Fetch again" runs it
again. Choosing a team only redraws. Nothing runs on a schedule.
"""

from __future__ import annotations

import streamlit as st

from nba_sos import live
from nba_sos.config import PROJECT_ROOT, Paths, load_configs

TABLE_COLS = {
    "team": "Team",
    "conference": "Conf",
    "wins_now": "W",
    "games_now": "GP",
    "exp_wins": "Exp. wins",
    "p05": "Wins p05",
    "p95": "Wins p95",
    "p_playoffs": "P(playoffs)",
    "p_top6": "P(top 6)",
    "exp_seed": "Exp. seed",
    "d_sos": "D-SOS",
    "d_sos_rank": "D-SOS rank",
    "S": "Strength",
    "sd": "Strength sd",
    "n_injured_out": "Players out",
}
REPORT_COLS = ["game_date", "matchup", "player", "status", "reason"]

st.set_page_config(page_title="NBA dynamic SOS: live projection", layout="wide")
st.title("NBA remaining-season projection")

paths = Paths(PROJECT_ROOT)
cfg = load_configs(paths)
if st.button("Fetch again") or "live" not in st.session_state:
    with st.spinner("Fetching schedule, results and the latest injury report, then simulating the season..."):
        st.session_state["live"] = live.run_live(paths, cfg)
res = st.session_state["live"]
meta, table = res["meta"], res["table"]

st.caption(
    f"Season {meta['season']}, snapshot {meta['snapshot']}, cutoff {meta['cutoff']}: "
    f"{meta['completed_games']} games played, {meta['remaining_scheduled_games']} remaining, "
    f"{meta['n_paths']} simulated seasons. Saved to {res['dir']}."
)
rep = meta["injury_report"]
st.caption(
    f"Official injury report {rep['path']} (published {rep['published_at_utc']}), shown for reference; "
    "the projection's availability terms come from game participation."
    if rep
    else "No official injury report found in the last 24 hours."
)

st.dataframe(table[list(TABLE_COLS)].rename(columns=TABLE_COLS).round(3), hide_index=True, width="stretch")

team = st.selectbox("Team", sorted(table.team))
row = table[table.team == team].iloc[0]
st.subheader(team)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected wins", f"{row.exp_wins:.1f}", help=f"90% interval {row.p05:.0f} to {row.p95:.0f}")
c2.metric("P(playoffs)", f"{row.p_playoffs:.1%}")
c3.metric("Remaining D-SOS", f"{row.d_sos:+.2f}", help=f"rank {int(row.d_sos_rank)} of {len(table)} (1 = hardest)")
c4.metric("Strength (per 100)", f"{row.S:+.1f}", help=f"sd {row.sd:.1f}")
st.dataframe(row.astype(str).to_frame("value"), width="stretch")  # mixed types: show as text
if res["injury_report"] is not None:
    r = res["injury_report"]
    st.markdown("**Official injury report**")
    st.dataframe(r[r.team_tricode == team][REPORT_COLS], hide_index=True, width="stretch")
