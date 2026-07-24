# ============================================================
# VIRTUAL VILLAGE — AI PRESENCE MONITORING
# POSTURE CLASSIFICATION + SMART ALERT SYSTEM
# UTD Capstone Project | Spring 2026
# ============================================================
# HOW IT WORKS:
#   1. Load sensor data from JSON
#   2. Train Random Forest on posture labels
#   3. Predict posture for every reading
#   4. Run alert logic on predictions
#   5. Live simulation
# ============================================================

import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ── SENSOR ROOM NAMES ────────────────────────────────────────
SENSOR_NAMES = {
    '0001': 'Living Room',
    '0002': 'Bedroom',
    '0003': 'Kitchen',
    '0004': 'Bathroom',
    '0005': 'Study',
    '0006': 'Hallway'
}

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
print('STEP 1 — LOADING DATA')
print('=' * 55)

import os

# ── SET YOUR DATA FILE PATH HERE ──────────────────────────────
# Option 1 (recommended): Place presence_data_4.json in the
#   same folder as this script — it will be found automatically
# Option 2: Paste your full path below and uncomment it
#   Windows : FILE = r'C:\Users\YourName\Desktop\presence_data_4.json'
#   Mac/Linux: FILE = '/Users/yourname/Downloads/presence_data_4.json'

FILE = r'C:\Users\selas\OneDrive\Documents\Presence Monitoring\Data\presence_data_4.json'

with open(FILE) as f:
    raw = json.load(f)

results = raw['data']['result']

COLS = ['x_pos', 'y_pos', 'z_height',
        'x_motion', 'y_motion', 'z_motion',
        'signal_strength', 'target_count']

records = []
for r in results:
    row = {
        'sensor_id': r['radarUuid'][-4:],
        'timestamp': datetime.strptime(
                     r['createTime'], '%Y%m%d%H%M%S')
    }
    for col, val in zip(COLS, r['parsedData']):
        row[col] = val
    records.append(row)

df = pd.DataFrame(records).reset_index(drop=True)
df = df.sort_values(
    ['sensor_id', 'timestamp']).reset_index(drop=True)
df['hour'] = df['timestamp'].dt.hour
df['date'] = df['timestamp'].dt.date
df['room'] = df['sensor_id'].map(SENSOR_NAMES)

print(f'✓ Loaded {len(df):,} records')
print(f'  Date range : {df["date"].min()} → {df["date"].max()}')
print(f'  Sensors    : {df["sensor_id"].nunique()} rooms')
print()

# ============================================================
# STEP 2 — CREATE POSTURE LABELS FROM Z HEIGHT
# Zones confirmed from actual data — not documentation
# Doc said min = 0.2m, actual data min = 0.05m
# ============================================================
print('STEP 2 — CREATING POSTURE LABELS')
print('=' * 55)

def get_posture(z):
    if 0.05 <= z <= 0.40:
        return 'Lying Down'   # sleeping / resting
    elif 0.41 <= z <= 0.80:
        return 'Sitting'      # TV / eating / relaxing
    elif 0.81 <= z <= 1.75:
        return 'Standing'     # walking / cooking
    else:
        return 'Unknown'

df['posture'] = df['z_height'].apply(get_posture)

# Numeric labels for model
posture_map = {
    'Lying Down': 0,
    'Sitting'   : 1,
    'Standing'  : 2,
    'Unknown'   : 3
}
label_map = {v: k for k, v in posture_map.items()}
df['posture_label'] = df['posture'].map(posture_map)

print('Posture zones:')
print('  0.05–0.40m → Lying Down')
print('  0.41–0.80m → Sitting')
print('  0.81–1.75m → Standing')
print()
for p, c in df['posture'].value_counts().items():
    pct = 100 * c / len(df)
    print(f'  {p:12s}: {c:>8,} ({pct:.1f}%)')
print()

# ============================================================
# STEP 3 — BALANCED SAMPLING + TRAIN MODEL
# Features: x_pos, y_pos, z_height, signal_strength
# Dropped: x_motion (0.99 corr), y_motion (0.99 corr),
#          z_motion (0.90 corr), target_count (always 1)
# Max 5,000 per sensor — avoids Living Room bias (71%)
# ============================================================
print('STEP 3 — TRAINING RANDOM FOREST')
print('=' * 55)

FEATURES = ['x_pos', 'y_pos',
            'z_height', 'signal_strength']

# Balanced sample — for loop (groupby causes KeyError)
sampled = []
for sensor in df['sensor_id'].unique():
    sub = df[df['sensor_id'] == sensor]
    n   = min(len(sub), 5000)
    sampled.append(sub.sample(n, random_state=42))

df_train = pd.concat(sampled).reset_index(drop=True)

print(f'Training on : {len(df_train):,} balanced records')
print(f'Features    : {FEATURES}')
print()

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1)

rf.fit(df_train[FEATURES],
       df_train['posture_label'])

print('✓ Model trained')
print()

# ============================================================
# STEP 4 — PREDICT ON ALL 160,297 RECORDS
# Note: wrap in pd.Series with index before .map()
# Direct .map() on numpy array raises AttributeError
# ============================================================
print('STEP 4 — PREDICTING ON ALL RECORDS')
print('=' * 55)
print(f'Running model on all {len(df):,} records...')

predictions = rf.predict(df[FEATURES])
df['pred_posture'] = pd.Series(
    predictions, index=df.index).map(label_map)
df['pred_confidence'] = rf.predict_proba(
    df[FEATURES]).max(axis=1)

print('✓ Predictions complete')
print()
for p, c in df['pred_posture'].value_counts().items():
    pct = 100 * c / len(df)
    print(f'  {p:12s}: {c:>8,} ({pct:.1f}%)')
print()

# ============================================================
# STEP 5 — SMART ALERT SYSTEM
# RF classifies posture → IF/ELSE checks context
# No second model needed
#
# Pattern 1: Lying during active hours (8am–8pm)
#            Person lying at 2pm = unusual → alert
#
# Pattern 2: Same posture 45+ minutes (active hours)
#            No movement for 45 mins during day → alert
# ============================================================
print('STEP 5 — SMART ALERT SYSTEM')
print('=' * 55)

# ── Pattern 1 ─────────────────────────────────
df['active_hour'] = df['hour'].between(8, 20)
df['alert_1']     = (
    (df['pred_posture'] == 'Lying Down') &
    df['active_hour'])
alert1 = df[df['alert_1']].copy()

print('PATTERN 1 — Lying During Active Hours (8am–8pm):')
print(f'  Total events : {len(alert1):,}')
print(f'  Peak hour    : {alert1["hour"].mode()[0]:02d}:00')
print(f'  Top room     : {alert1["room"].mode()[0]}')
print(f'  → VV Speaker : "Are you ok?"')
print()

# ── Pattern 2 ─────────────────────────────────
alerts_2 = []
for sensor in df['sensor_id'].unique():
    sub = df[df['sensor_id'] == sensor].copy()
    sub = sub.sort_values('timestamp')
    sub['changed'] = (
        sub['pred_posture'] !=
        sub['pred_posture'].shift(1))
    sub['group'] = sub['changed'].cumsum()
    for _, grp in sub.groupby('group'):
        posture  = grp['pred_posture'].iloc[0]
        start    = grp['timestamp'].iloc[0]
        end      = grp['timestamp'].iloc[-1]
        duration = (end - start).total_seconds() / 60
        hour     = start.hour
        if (duration >= 45 and
                8 <= hour <= 20 and
                posture in ['Lying Down','Sitting']):
            alerts_2.append({
                'date'    : start.date(),
                'room'    : grp['room'].iloc[0],
                'posture' : posture,
                'duration': round(duration, 1),
                'hour'    : hour
            })

df_a2 = pd.DataFrame(alerts_2)

print('PATTERN 2 — Same Posture 45+ Minutes:')
print(f'  Total events : {len(df_a2):,}')
if len(df_a2) > 0:
    print(f'  Longest      : {df_a2["duration"].max():.0f} mins')
print(f'  → VV Speaker : "Are you ok?"')
print()

# ============================================================
# STEP 6 — LIVE SIMULATION
# Shows how system processes each sensor reading
# in real deployment (every 7 seconds)
# ============================================================
print('STEP 6 — LIVE SYSTEM SIMULATION')
print('=' * 55)
print('Simulating real-time sensor readings...')
print('(2 second delay between each reading)')
print()

# Mix: normal readings + alert readings
sitting   = df[(df['sensor_id']=='0001') &
               (df['hour'].between(9,11)) &
               (df['pred_posture']=='Sitting')].head(2)
standing  = df[(df['sensor_id']=='0001') &
               (df['hour'].between(9,11)) &
               (df['pred_posture']=='Standing')].head(2)
alert_r   = df[(df['sensor_id']=='0001') &
               (df['hour'].between(13,16)) &
               (df['pred_posture']=='Lying Down')].head(2)
sleeping  = df[(df['sensor_id']=='0002') &
               (df['hour'].between(0,5)) &
               (df['pred_posture']=='Lying Down')].head(2)

live = pd.concat([
    sitting, standing, alert_r, sleeping
]).reset_index(drop=True)

print("""
╔══════════════════════════════════════════════════════════╗
║     VIRTUAL VILLAGE — LIVE MONITORING STARTED           ║
╚══════════════════════════════════════════════════════════╝
""")

for i, row in live.iterrows():
    posture    = row['pred_posture']
    confidence = row['pred_confidence'] * 100
    hour       = row['hour']
    room       = row['room']
    z          = row['z_height']
    ts         = row['timestamp']
    alert      = (posture == 'Lying Down'
                  and 8 <= hour <= 20)

    print('━'*52)
    print('  🚨 ALERT TRIGGERED' if alert
          else '  ✅ NORMAL READING')
    print('━'*52)
    print(f'  Timestamp  : {ts}')
    print(f'  Room       : {room}')
    print(f'  Z Height   : {z:.3f}m')
    print()
    print(f'  RANDOM FOREST → {posture}')
    print(f'  Confidence   : {confidence:.1f}%')
    print()

    if alert:
        print(f'  Hour {hour:02d}:00 is active hours (8am–8pm)')
        print(f'  Lying at this hour = unusual')
        print(f'  🔊 VV SPEAKER : "Are you ok?"')
        print(f'  📱 FAMILY APP : Notification sent')
        print(f'  🚑 NO RESPONSE: 911 contacted')
    elif posture == 'Lying Down':
        print(f'  Hour {hour:02d}:00 = sleeping hours → normal')
    elif posture == 'Sitting':
        print(f'  Sitting at {hour:02d}:00 → normal')
    elif posture == 'Standing':
        print(f'  Standing at {hour:02d}:00 → normal')

    print()
    time.sleep(2)

print('━'*52)
print(f"""
╔══════════════════════════════════════════════════════════╗
║  DONE                                                    ║
║                                                          ║
║  Model    : Random Forest (100 trees, depth 15)         ║
║  Features : x_pos, y_pos, z_height, signal_strength     ║
║  Labels   : Lying Down / Sitting / Standing / Unknown   ║
║                                                          ║
║  Alert 1  : {len(alert1):,} events — lying during 8am–8pm        ║
║  Alert 2  : {len(df_a2):,} events — 45+ min same posture          ║
╚══════════════════════════════════════════════════════════╝
""")
