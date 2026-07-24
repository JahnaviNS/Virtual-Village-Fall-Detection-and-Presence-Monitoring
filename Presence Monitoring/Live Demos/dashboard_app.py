# ════════════════════════════════════════════════════════════════
# VIRTUAL VILLAGE — STREAMLIT LIVE DASHBOARD
# Environment : presence_env
# Run command : c:\ProgramData\anaconda3\envs\presence_env\python.exe -m streamlit run dashboard_app.py
# ════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Virtual Village — Live Monitor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .alert-box {
        background: #fff0f0;
        border-left: 4px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 6px 0;
    }
    .normal-box {
        background: #f0fff4;
        border-left: 4px solid #2ecc71;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 6px 0;
    }
    .sleep-box {
        background: #f0f8ff;
        border-left: 4px solid #3498db;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 6px 0;
    }
    .room-title { font-weight: 700; font-size: 0.95rem; }
    .room-meta  { font-size: 0.78rem; color: #666; margin-top: 4px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "sensor_data.csv")
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error("sensor_data.csv not found. Make sure it is in the same folder as this file.")
        st.stop()

df = load_data()

# ── Debug panel ───────────────────────────────────────────────
with st.expander("Debug — inspect loaded data (collapse when done)"):
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Posture values:**",
             df['pred_posture'].unique() if 'pred_posture' in df.columns else "Column not found")
    st.write("**Sensor IDs:**",
             df['sensor_id'].unique() if 'sensor_id' in df.columns else "Column not found")
    st.write("**Hour range:**",
             df['hour'].min() if 'hour' in df.columns else "N/A",
             "to",
             df['hour'].max() if 'hour' in df.columns else "N/A")
    st.write("**Rooms:**",
             df['room'].unique() if 'room' in df.columns else "Column not found")
    st.write("**Shape:**", df.shape)
    st.dataframe(df.head(10), use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Settings")
st.sidebar.divider()

alert_hour_start = st.sidebar.slider("Alert window start (hour)", 0, 12, 8)
alert_hour_end   = st.sidebar.slider("Alert window end (hour)", 12, 23, 20)
refresh_speed    = st.sidebar.selectbox("Refresh every (seconds)", [1, 2, 5], index=1)

sensor_options = ["All"]
if 'sensor_id' in df.columns:
    sensor_options += sorted(df['sensor_id'].unique().tolist())
selected_sensor = st.sidebar.selectbox("Filter by Sensor", sensor_options)

show_raw = st.sidebar.checkbox("Show raw data table")

st.sidebar.divider()
st.sidebar.markdown("**Model Info**")
st.sidebar.info(
    "Model: Random Forest\n\n"
    "Features: x_pos, y_pos, z_height, signal\n\n"
    "Classes: Sitting / Standing / Lying Down"
)
st.sidebar.divider()
st.sidebar.caption("Virtual Village Presence Monitoring System")


# ── Build simulation feed ─────────────────────────────────────
def build_live_feed(df, sensor_filter):
    filtered = df.copy()

    if sensor_filter != "All" and 'sensor_id' in filtered.columns:
        filtered = filtered[filtered['sensor_id'] == sensor_filter]

    parts = []
    for posture in ['Sitting', 'Standing', 'Lying Down']:
        if 'pred_posture' in filtered.columns:
            subset = filtered[filtered['pred_posture'] == posture].head(3)
            if len(subset) > 0:
                parts.append(subset)

    if parts:
        live = pd.concat(parts).reset_index(drop=True)
    else:
        live = filtered.head(12).reset_index(drop=True)

    if len(live) == 0:
        st.error(
            "No data rows found. "
            "Check the debug panel above for column names and values."
        )
        st.stop()

    return live

live = build_live_feed(df, selected_sensor)


# ── Session state ─────────────────────────────────────────────
defaults = {
    'reading_idx': 0,
    'z_history':   [],
    'alert_log':   [],
    'total':       0,
    'alert_count': 0,
    'running':     False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Header ────────────────────────────────────────────────────
st.title("🏠 Virtual Village — Live Monitor")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── Control buttons ───────────────────────────────────────────
b1, b2, b3 = st.columns([1, 1, 6])
if b1.button("▶ Start", use_container_width=True, type="primary"):
    st.session_state.running = True
if b2.button("⏹ Stop", use_container_width=True):
    st.session_state.running = False

st.divider()

# ── Stat cards ────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Readings",   st.session_state.total)
c2.metric("Alerts Triggered", st.session_state.alert_count)
c3.metric("Active Sensors",   df['sensor_id'].nunique() if 'sensor_id' in df.columns else "N/A")
c4.metric("Rooms Monitored",  df['room'].nunique()      if 'room'      in df.columns else "N/A")

st.divider()

# ── Main layout ───────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.subheader("Live z_height Feed — Last 60 Readings")
    chart_placeholder = st.empty()
    st.subheader("Alert Log")
    log_placeholder = st.empty()

with right:
    st.subheader("Current Reading")
    room_placeholder = st.empty()

# ── Raw data (optional) ───────────────────────────────────────
if show_raw:
    st.divider()
    st.subheader("Raw Sensor Data")
    st.dataframe(df, use_container_width=True)


# ── Live loop ─────────────────────────────────────────────────
if st.session_state.running and len(live) > 0:

    idx = st.session_state.reading_idx % len(live)
    row = live.iloc[idx]

    posture    = row.get('pred_posture',    'Unknown')
    confidence = float(row.get('pred_confidence', 0)) * 100
    hour       = int(row.get('hour', 12))
    room       = row.get('room', 'Unknown Room')
    z          = float(row.get('z_height', 1.0))
    ts         = str(row.get('timestamp', datetime.now()))
    sensor_id  = row.get('sensor_id', 'N/A')

    is_alert = (posture == 'Lying Down' and
                alert_hour_start <= hour <= alert_hour_end)

    st.session_state.z_history.append(z)
    if len(st.session_state.z_history) > 60:
        st.session_state.z_history.pop(0)

    st.session_state.total += 1
    st.session_state.reading_idx += 1

    if is_alert:
        st.session_state.alert_count += 1
        st.session_state.alert_log.insert(0, {
            "Time":    ts,
            "Room":    room,
            "Posture": posture,
            "z (m)":   round(z, 3),
            "Conf %":  round(confidence, 1),
            "Sensor":  sensor_id,
            "Status":  "ALERT"
        })
        if len(st.session_state.alert_log) > 20:
            st.session_state.alert_log.pop()

    # ── Chart ─────────────────────────────────────────────────
    with chart_placeholder:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=st.session_state.z_history,
            mode='lines',
            line=dict(color='#4488ff', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(68,136,255,0.08)',
            name='z_height'
        ))
        fig.add_hline(
            y=0.4,
            line_dash='dash',
            line_color='red',
            annotation_text='Fall threshold (0.4m)',
            annotation_position='bottom right'
        )
        fig.update_layout(
            yaxis=dict(range=[0, 2.2], title='z_height (m)'),
            xaxis=dict(title='Reading number'),
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        st.plotly_chart(fig, use_container_width=True)

    # ── Room card ─────────────────────────────────────────────
    with room_placeholder:
        if is_alert:
            box_cls  = "alert-box"
            badge    = "ALERT"
            decision = "UNUSUAL — Lying Down during active hours"
        elif posture == 'Lying Down':
            box_cls  = "sleep-box"
            badge    = "Sleeping"
            decision = "Normal — sleeping hours"
        elif posture == 'Sitting':
            box_cls  = "normal-box"
            badge    = "Normal"
            decision = "Normal — resting"
        else:
            box_cls  = "normal-box"
            badge    = "Normal"
            decision = "Normal — active"

        st.markdown(f"""
        <div class="{box_cls}">
            <div class="room-title">
                {room} &nbsp;
                <small style="color:#aaa; font-weight:400">{sensor_id}</small>
            </div>
            <div class="room-meta">
                Posture &nbsp;&nbsp;: <b>{posture}</b><br>
                z_height &nbsp;: <b>{z:.3f} m</b><br>
                Confidence : <b>{confidence:.1f}%</b><br>
                Hour &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>{hour:02d}:00</b><br>
                Timestamp : <b>{ts}</b>
            </div>
            <div style="margin-top:8px; font-size:0.82rem; font-weight:600">
                {badge} — {decision}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if is_alert:
            st.error("Speaker: Are you ok?")
            st.warning("Family App: Notification sent")

    # ── Alert log ─────────────────────────────────────────────
    with log_placeholder:
        if st.session_state.alert_log:
            log_df = pd.DataFrame(st.session_state.alert_log)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No alerts yet...")

    time.sleep(refresh_speed)
    st.rerun()

# ── Idle state ────────────────────────────────────────────────
elif not st.session_state.running:
    with chart_placeholder:
        st.info("Press Start to begin live monitoring.")
    with room_placeholder:
        st.caption("Waiting for data...")
    with log_placeholder:
        st.caption("No alerts yet...")


#Using th following command in the terminal to run the streamlit app:
