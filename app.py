"""
Interactive Digital Twin Replay Dashboard
==========================================
Run locally with:  streamlit run app.py

Requires `dashboard_data.json` (produced by Prepare_DashboardData.ipynb)
to be in the SAME FOLDER as this file.

Install requirements first:
    pip install streamlit plotly numpy
"""

import json
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Disaster Response Digital Twin", layout="wide")

# ---------------------------------------------------------------------------
# Load precomputed data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    with open("dashboard_data.json", "r") as f:
        return json.load(f)

try:
    data = load_data()
except FileNotFoundError:
    st.error(
        "dashboard_data.json not found. Run Prepare_DashboardData.ipynb in Colab, "
        "download the resulting file from Google Drive, and place it in this same folder as app.py."
    )
    st.stop()

meta = data["meta"]
n_timesteps = meta["n_timesteps"]
extent = meta["extent"]  # [left, right, bottom, top]
flood_small = np.array(data["flood_raster_small"])
road_edges = data["road_edges"]
resources = data["resources"]
policies = data["policies"]

RESOURCE_COLORS = {"hospital": "#1f77b4", "shelter": "#ff7f0e", "ambulance_depot": "#9467bd"}
RESOURCE_SYMBOLS = {"hospital": "triangle-up", "shelter": "square", "ambulance_depot": "diamond"}

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Digital Twin Controls")
st.sidebar.markdown(
    "This dashboard **replays** a precomputed disaster-response simulation. "
    "It is not connected to a live sensor feed -- consistent with the project's "
    "scope (simulated/historical data replay)."
)

policy_name = st.sidebar.selectbox("Policy", list(policies.keys()))
frames = policies[policy_name]

if "timestep_idx" not in st.session_state:
    st.session_state.timestep_idx = 0

autoplay = st.sidebar.checkbox("Auto-play", value=False)

col_a, col_b, col_c = st.sidebar.columns(3)
if col_a.button("⏮ Reset"):
    st.session_state.timestep_idx = 0
if col_b.button("◀ Prev"):
    st.session_state.timestep_idx = max(0, st.session_state.timestep_idx - 1)
if col_c.button("Next ▶"):
    st.session_state.timestep_idx = min(len(frames) - 1, st.session_state.timestep_idx + 1)

timestep_idx = st.sidebar.slider("Hour", 0, len(frames) - 1, st.session_state.timestep_idx)
st.session_state.timestep_idx = timestep_idx

frame = frames[timestep_idx]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Legend**\n"
    "- 🔴 Red: pending (needs help)\n"
    "- 🟢 Green: resolved (helped)\n"
    "- 🔺 Triangle: hospital\n"
    "- ◼️ Square: shelter\n"
    "- 🔷 Diamond: ambulance depot"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌊 Chennai Flood Response — Digital Twin Simulation")
st.markdown(
    f"**Policy:** {policy_name}  |  **Hour:** {frame['timestep']} / {n_timesteps}"
)

# ---------------------------------------------------------------------------
# Map figure
# ---------------------------------------------------------------------------
fig = go.Figure()

# Flood severity heatmap (downsampled background)
fig.add_trace(go.Heatmap(
    z=flood_small,
    x=np.linspace(extent[0], extent[1], flood_small.shape[1]),
    y=np.linspace(extent[3], extent[2], flood_small.shape[0]),
    colorscale="Blues",
    opacity=0.55,
    showscale=False,
    hoverinfo="skip",
))

# Road network sketch (sampled edges)
for edge in road_edges:
    fig.add_trace(go.Scatter(
        x=edge["lons"], y=edge["lats"],
        mode="lines",
        line=dict(color="lightgray", width=0.5),
        hoverinfo="skip",
        showlegend=False,
    ))

# Pending requests
fig.add_trace(go.Scatter(
    x=frame["pending_lons"], y=frame["pending_lats"],
    mode="markers",
    marker=dict(color="red", size=4, opacity=0.5),
    name=f"Pending ({frame['n_pending']:,})",
))

# Resolved requests (accumulated)
fig.add_trace(go.Scatter(
    x=frame["resolved_lons"], y=frame["resolved_lats"],
    mode="markers",
    marker=dict(color="limegreen", size=6, opacity=0.8, line=dict(color="darkgreen", width=0.5)),
    name=f"Resolved so far ({frame['n_resolved_total']:,})",
))

# Resource markers
for rtype, color in RESOURCE_COLORS.items():
    subset = [r for r in resources if r["type"] == rtype]
    fig.add_trace(go.Scatter(
        x=[r["lon"] for r in subset], y=[r["lat"] for r in subset],
        mode="markers",
        marker=dict(color=color, size=9, symbol=RESOURCE_SYMBOLS[rtype],
                    line=dict(color="black", width=0.5)),
        name=rtype.replace("_", " ").title(),
    ))

fig.update_layout(
    height=650,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=False, zeroline=False, title="Longitude"),
    yaxis=dict(showgrid=False, zeroline=False, title="Latitude", scaleanchor="x"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    plot_bgcolor="white",
)

# ---------------------------------------------------------------------------
# Layout: map + live stats side by side
# ---------------------------------------------------------------------------
col_map, col_stats = st.columns([2.2, 1])

with col_map:
    st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.subheader("Live Stats")
    st.metric("Pending Requests", f"{frame['n_pending']:,}")
    st.metric("Total Resolved So Far", f"{frame['n_resolved_total']:,}")

    total_seen = frame["n_pending"] + frame["n_resolved_total"]
    pct_resolved = (frame["n_resolved_total"] / total_seen * 100) if total_seen > 0 else 0
    st.progress(min(pct_resolved / 100, 1.0))
    st.caption(f"{pct_resolved:.1f}% of requests seen so far have been resolved")

    st.markdown("**Resource Utilization**")
    util = frame["utilization"]
    for rtype in ["hospital", "shelter", "ambulance_depot"]:
        used, total = util[rtype]["used"], util[rtype]["total"]
        pct = (used / total * 100) if total > 0 else 0
        st.write(f"{rtype.replace('_', ' ').title()}: {used} / {total} ({pct:.1f}%)")
        st.progress(min(pct / 100, 1.0))

# ---------------------------------------------------------------------------
# Auto-play loop
# ---------------------------------------------------------------------------
if autoplay and st.session_state.timestep_idx < len(frames) - 1:
    import time
    time.sleep(1.0)
    st.session_state.timestep_idx += 1
    st.rerun()

st.markdown("---")
st.caption(
    "This dashboard replays a precomputed simulation (Digital Twin + RL/rule-based/random policy). "
    "It demonstrates the framework's behavior over a full simulated flood event and is not connected "
    "to a live external data feed in the current project scope."
)
