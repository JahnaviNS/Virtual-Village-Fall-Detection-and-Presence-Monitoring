# ════════════════════════════════════════════════════════════════
# VIRTUAL VILLAGE — LIVE ALERT DEMO
# Scenario: Person in Living Room detected Lying Down at 2pm
# This triggers:
#   Pattern 1 — Lying during active hours (8am–8pm)
#   Pattern 2 — Same posture 45+ mins during active hours
#
# HOW THE DATA WAS CONSTRUCTED:
#   - Rows 1–5  : Normal morning activity (Sitting/Standing, 9–11am)
#   - Row 6     : First alert — Lying Down at 14:00 (2pm) → Pattern 1
#   - Rows 7–11 : Continued Lying Down at 14:05–14:32 (still unusual)
#   - Row 12    : Pattern 2 triggered — same posture 45+ mins
#   - Rows 13–15: Person wakes up, returns to normal
#
# WHY THIS IS REALISTIC:
#   From your Step 8 output: 655 Pattern 1 events peak at 11:00
#   z_height 0.05–0.40m confirms Lying Down per your posture zones
#   All confidence scores match your model's 100% output pattern
#   Sensor 0001 = Living Room (your most active sensor, 71.4% of data)
#
# Run:
#   c:\ProgramData\anaconda3\envs\presence_env\python.exe
#   -m streamlit run alert_demo.py
# ════════════════════════════════════════════════════════════════

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import time
from datetime import datetime

st.set_page_config(
    page_title="VV — Live Alert Demo",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Scenario feed ─────────────────────────────────────────────
# (room, sensor, posture, z_height, confidence, hour, timestamp,
#  is_alert, alert_type, note)
FEED = [
    # Normal morning readings
    ('Living Room','0001','Sitting',   0.779,100.0, 9,'2026-03-09 09:12:00',
     False,'','Normal — resting after breakfast'),
    ('Living Room','0001','Standing',  0.921,100.0, 9,'2026-03-09 09:18:42',
     False,'','Normal — moving around'),
    ('Kitchen',    '0003','Standing',  0.934,100.0,10,'2026-03-09 10:05:17',
     False,'','Normal — cooking/preparing'),
    ('Living Room','0001','Sitting',   0.689,100.0,11,'2026-03-09 11:42:05',
     False,'','Normal — watching TV'),
    ('Living Room','0001','Standing',  0.829,100.0,11,'2026-03-09 11:58:33',
     False,'','Normal — active'),

    # ── ALERT BEGINS: Lying Down at 2pm ──────────────────────
    ('Living Room','0001','Lying Down',0.228,100.0,14,'2026-03-09 14:02:11',
     True, 'Pattern 1',
     'UNUSUAL — Lying Down at 14:00 (active hours 8am–8pm)'),
    ('Living Room','0001','Lying Down',0.195,100.0,14,'2026-03-09 14:09:18',
     True, 'Pattern 1',
     'UNUSUAL — Still lying down 7 mins later'),
    ('Living Room','0001','Lying Down',0.211,100.0,14,'2026-03-09 14:16:44',
     True, 'Pattern 1',
     'UNUSUAL — Still lying down 14 mins in'),
    ('Living Room','0001','Lying Down',0.203,100.0,14,'2026-03-09 14:23:55',
     True, 'Pattern 1',
     'UNUSUAL — 21 mins, no movement detected'),
    ('Living Room','0001','Lying Down',0.218,100.0,14,'2026-03-09 14:31:02',
     True, 'Pattern 1',
     'UNUSUAL — 29 mins, still no response'),
    ('Living Room','0001','Lying Down',0.199,100.0,14,'2026-03-09 14:38:49',
     True, 'Pattern 1',
     'UNUSUAL — 36 mins, posture unchanged'),

    # ── Pattern 2 triggers at 45+ mins ───────────────────────
    ('Living Room','0001','Lying Down',0.207,100.0,14,'2026-03-09 14:47:30',
     True, 'Pattern 1 + Pattern 2',
     'CRITICAL — 45+ mins same posture! Both patterns triggered'),

    # Person eventually gets up
    ('Living Room','0001','Sitting',   0.651,100.0,15,'2026-03-09 15:12:00',
     False,'','Person responded — now sitting'),
    ('Living Room','0001','Standing',  0.876,100.0,15,'2026-03-09 15:18:22',
     False,'','Normal — person active again'),
    ('Kitchen',    '0003','Standing',  0.912,100.0,15,'2026-03-09 15:24:45',
     False,'','Normal — moved to kitchen'),
]

# ── Session state ─────────────────────────────────────────────
for k,v in {
    'idx':0,'z_hist':[],'log':[],'total':0,
    'alerts':0,'p2_triggered':False,'running':False
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────
st.title("🏠 Virtual Village — Live Alert Demo")
st.caption(
    "Scenario: Person detected Lying Down at 2pm in Living Room — "
    "sustained for 45+ minutes  |  "
    f"Updated: {datetime.now().strftime('%H:%M:%S')}"
)

# ── Scenario explanation box ──────────────────────────────────
with st.expander("About this scenario — Why these readings trigger alerts"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**How the data was constructed:**
- Rows 1–5: Normal morning activity (Sitting/Standing, 9–11am)
- Row 6: First alert — Lying Down detected at **14:00 (2pm)**
- Rows 7–12: Continued Lying Down readings every ~7 seconds
- Row 12: **45+ minutes elapsed** → Pattern 2 also triggered
- Rows 13–15: Person eventually responds and gets up

**Source of z_height values:**
- All Lying Down readings use z = 0.19–0.23m
- This is within your model's confirmed zone: **0.05–0.40m = Lying Down**
- Sensor 0001 = Living Room — your highest activity sensor (71.4% of data)
        """)
    with col2:
        st.markdown("""
**Why this triggers alerts — from your Step 8 logic:**

**Pattern 1 trigger condition:**
```python
is_alert = (posture == 'Lying Down' and
            8 <= hour <= 20)
```
→ hour = 14 (2pm), posture = Lying Down ✓

**Pattern 2 trigger condition:**
```python
if duration >= 45 and 8 <= hour <= 20
   and posture in ['Lying Down', 'Sitting']:
```
→ Same posture held from 14:02 to 14:47 = 45 mins ✓

**From your actual Step 8 output:**
Pattern 1 found 655 events — peak at 11:00, top room: Living Room
Pattern 2 found 14 events — longest: 2,829 mins
        """)

# ── Control buttons ───────────────────────────────────────────
b1, b2, b3 = st.columns([1, 1, 5])
if b1.button("▶ Start", type="primary", use_container_width=True):
    st.session_state.running = True
if b2.button("⏹ Stop", use_container_width=True):
    st.session_state.running = False

st.divider()

# ── Stat cards ────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Readings",    st.session_state.total)
c2.metric("Alerts Triggered",  st.session_state.alerts)
c3.metric("Pattern 2 Status",
          "TRIGGERED" if st.session_state.p2_triggered else "Monitoring")
c4.metric("Model Accuracy",    "100.00%")

st.divider()

# ── Layout ────────────────────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.subheader("Live z_height Feed")
    chart_ph = st.empty()
    st.subheader("Alert Log")
    log_ph = st.empty()

with right:
    st.subheader("Current Reading")
    card_ph = st.empty()
    st.subheader("System Decision")
    decision_ph = st.empty()

# ── Live loop ─────────────────────────────────────────────────
if st.session_state.running:

    if st.session_state.idx >= len(FEED):
        # Reset
        st.session_state.idx          = 0
        st.session_state.total        = 0
        st.session_state.alerts       = 0
        st.session_state.z_hist       = []
        st.session_state.log          = []
        st.session_state.p2_triggered = False

    room, sensor, posture, z, conf, hour, ts, is_alert, atype, note = \
        FEED[st.session_state.idx]

    st.session_state.z_hist.append(z)
    st.session_state.total += 1
    st.session_state.idx   += 1

    if is_alert:
        st.session_state.alerts += 1
        if 'Pattern 2' in atype:
            st.session_state.p2_triggered = True

        st.session_state.log.insert(0, {
            "Timestamp": ts,
            "Room":      room,
            "Posture":   posture,
            "z (m)":     z,
            "Hour":      f"{hour:02d}:00",
            "Alert":     atype,
            "Note":      note
        })

    # ── Chart ─────────────────────────────────────────────────
    with chart_ph:
        # Colour each point
        colors = []
        for i in range(len(st.session_state.z_hist)):
            row_data = FEED[i]
            if 'Pattern 2' in row_data[8]:
                colors.append('#7C3AED')   # purple = Pattern 2
            elif row_data[7]:
                colors.append('#DC2626')   # red = Pattern 1
            else:
                colors.append('#2563EB')   # blue = normal

        fig = go.Figure()

        # z_height line
        fig.add_trace(go.Scatter(
            y=st.session_state.z_hist,
            mode='lines+markers',
            line=dict(color='#2563EB', width=1.5),
            marker=dict(size=7, color=colors),
            fill='tozeroy',
            fillcolor='rgba(37,99,235,0.06)',
            name='z_height'
        ))

        # Fall / Lying Down threshold
        fig.add_hline(
            y=0.40, line_dash='dash', line_color='#DC2626',
            annotation_text='Lying Down threshold (0.40m)',
            annotation_position='bottom right'
        )

        # Shade the alert zone
        if len(st.session_state.z_hist) > 5:
            alert_indices = [i for i,f in enumerate(FEED[:len(st.session_state.z_hist)]) if f[7]]
            if alert_indices:
                fig.add_vrect(
                    x0=alert_indices[0]-0.5,
                    x1=len(st.session_state.z_hist)-0.5,
                    fillcolor='rgba(220,38,38,0.07)',
                    line_width=0,
                    annotation_text="Alert zone",
                    annotation_position="top left"
                )

        fig.update_layout(
            yaxis=dict(range=[0, 1.6], title='z_height (m)'),
            xaxis=dict(title='Reading number'),
            height=270,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=True, gridcolor='#F1F5F9')
        fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9')
        st.plotly_chart(fig, use_container_width=True)

    # ── Current reading card ───────────────────────────────────
    with card_ph:
        if 'Pattern 2' in atype:
            st.error(f"**{room}**  ·  Sensor {sensor}")
        elif is_alert:
            st.warning(f"**{room}**  ·  Sensor {sensor}")
        else:
            st.success(f"**{room}**  ·  Sensor {sensor}")

        st.markdown(f"""
| Field | Value |
|---|---|
| Timestamp | `{ts}` |
| Posture | **{posture}** |
| z_height | **{z:.3f} m** |
| Confidence | **{conf:.1f}%** |
| Hour | **{hour:02d}:00** |
        """)

    # ── Decision panel ────────────────────────────────────────
    with decision_ph:
        if 'Pattern 2' in atype:
            st.error("**PATTERN 1 + PATTERN 2 TRIGGERED**")
            st.error("Lying Down since 14:02 — now 45+ mins")
            st.warning("🔊 Speaker: Are you ok?")
            st.warning("📱 Family App: Notification sent")
            st.error("🚑 No response → 911 contacted")
        elif is_alert:
            st.warning(f"**{atype} TRIGGERED**")
            st.warning(note)
            st.info("🔊 Speaker: Are you ok?")
            st.info("📱 Family App: Notification sent")
        else:
            st.success("✅ Normal")
            st.caption(note)

    # ── Alert log ─────────────────────────────────────────────
    with log_ph:
        if st.session_state.log:
            st.dataframe(
                pd.DataFrame(st.session_state.log),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption("No alerts yet — normal activity...")

    time.sleep(2)
    st.rerun()

elif not st.session_state.running:
    with chart_ph:
        st.info(
            "Press **Start** to run the scenario.\n\n"
            "The feed shows normal morning activity first, "
            "then a person lying down at 2pm triggers both alert patterns."
        )
    with card_ph:
        st.caption("Waiting...")
    with decision_ph:
        st.caption("")
    with log_ph:
        st.caption("No alerts yet...")
