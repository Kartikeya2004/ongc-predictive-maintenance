import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="ONGC Predictive Maintenance", layout="wide")

# ── Load & prepare data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    columns = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + \
              [f'sensor{i}' for i in range(1, 22)]
    train = pd.read_csv('train_FD001.txt', sep='\s+', header=None, names=columns)
    max_cycles = train.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    train = train.merge(max_cycles, on='engine_id')
    train['RUL'] = train['max_cycle'] - train['cycle']
    train.drop(columns=['max_cycle'], inplace=True)

    drop_cols = ['sensor1','sensor5','sensor6','sensor8','sensor10',
                 'sensor13','sensor15','sensor16','sensor18','sensor19',
                 'sensor20','sensor21','setting1','setting2','setting3']
    train = train.drop(columns=drop_cols)
    train['failure_risk'] = (train['RUL'] <= 30).astype(int)
    return train

@st.cache_resource
def train_model(train):
    feature_cols = [c for c in train.columns if c not in ['engine_id','RUL','failure_risk']]
    X = train[feature_cols]
    y = train['failure_risk']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=17531/3100, random_state=42,
                          eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    return model, scaler, feature_cols

# ── Train Isolation Forest on healthy data (RUL > 60) ────────────────
@st.cache_resource
def train_anomaly_detectors(train, feature_cols):
    """
    Train one Isolation Forest per sensor using only healthy data (RUL > 60).
    This teaches the model what 'normal' looks like for each sensor.
    Any reading that deviates from this normal baseline is flagged as anomalous.
    """
    healthy = train[train['RUL'] > 60]
    detectors = {}
    for sensor in feature_cols:
        if sensor == 'cycle':
            continue
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(healthy[[sensor]])
        detectors[sensor] = iso
    return detectors

train = load_data()
model, scaler, feature_cols = train_model(train)
sensor_cols = [c for c in feature_cols if c != 'cycle']
detectors = train_anomaly_detectors(train, feature_cols)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/8/8d/ONGC_Logo.svg/200px-ONGC_Logo.svg.png",
    width=150
)
st.sidebar.title("ONGC Predictive Maintenance")
st.sidebar.markdown("AI-powered equipment failure detection system")

engine_id = st.sidebar.selectbox(
    "Select Engine / Equipment ID",
    sorted(train['engine_id'].unique())
)

life_pct = st.sidebar.slider(
    "⏱️ Equipment Life (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    help="0% = brand new  |  100% = end of life."
)

st.sidebar.caption(
    "📌 This slider is relative — 50% means each equipment is at "
    "the midpoint of *its own* lifecycle, regardless of its absolute cycle count."
)

st.sidebar.divider()
st.sidebar.info("📋 Maintenance Schedule (bottom of page) uses the reading at RUL ≥ 60 per equipment — independent of the slider.")

# ── Resolve selected engine row from life % ───────────────────────────
engine_data = train[train['engine_id'] == engine_id].copy().reset_index(drop=True)
total_rows = len(engine_data)
row_idx = min(int(total_rows * life_pct / 100), total_rows - 1)
row = engine_data.iloc[row_idx]

selected_cycle = int(row['cycle'])
max_cycle = int(engine_data['cycle'].max())
current_rul = int(row['RUL'])

risk_prob = model.predict_proba(
    scaler.transform(engine_data[feature_cols].iloc[[row_idx]])
)[0][1]

# ── Main Header ───────────────────────────────────────────────────────
st.title("🛢️ ONGC Equipment Health Monitor")
st.markdown("Real-time failure risk prediction using Machine Learning")
st.divider()

# ── Metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Equipment ID", f"EQ-{engine_id:03d}")
col2.metric("Life Stage", f"{life_pct}%")
col3.metric("Actual Cycle", selected_cycle)
col4.metric("Remaining Useful Life", f"{current_rul} cycles")
col5.metric("Failure Risk", f"{risk_prob*100:.1f}%",
            delta="⚠️ High Risk" if risk_prob > 0.5 else "✅ Normal",
            delta_color="inverse")

st.divider()

# ── Risk Alert ────────────────────────────────────────────────────────
if risk_prob > 0.7:
    st.error(
        f"🚨 CRITICAL: Equipment EQ-{engine_id:03d} has {risk_prob*100:.1f}% failure "
        f"probability at {life_pct}% of its life (cycle {selected_cycle}). "
        f"Immediate maintenance required!"
    )
elif risk_prob > 0.4:
    st.warning(
        f"⚠️ WARNING: Equipment EQ-{engine_id:03d} showing early failure signs "
        f"at {life_pct}% of its life (cycle {selected_cycle}). Schedule maintenance soon."
    )
else:
    st.success(
        f"✅ Equipment EQ-{engine_id:03d} is operating normally "
        f"at {life_pct}% of its life (cycle {selected_cycle})."
    )

# ── Sensor Trends ─────────────────────────────────────────────────────
st.subheader("📈 Sensor Degradation Trends")
top_sensors = ['sensor11', 'sensor4', 'sensor9']
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, sensor in zip(axes, top_sensors):
    ax.plot(engine_data['cycle'], engine_data[sensor], color='steelblue', linewidth=1.2)
    ax.axvline(x=selected_cycle, color='red', linestyle='--',
               label=f'Cycle {selected_cycle} ({life_pct}% life)')
    ax.set_title(f'{sensor.upper()} Reading')
    ax.set_xlabel('Cycle')
    ax.legend(fontsize=8)
plt.tight_layout()
st.pyplot(fig)

# ── Risk Over Time ────────────────────────────────────────────────────
st.subheader("📊 Failure Risk Probability Over Time")
probs = model.predict_proba(
    scaler.transform(engine_data[feature_cols])
)[:, 1]

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.plot(engine_data['cycle'].values, probs, color='crimson', linewidth=1.5)
ax2.axvline(x=selected_cycle, color='blue', linestyle='--', linewidth=2,
            label=f'Current: Cycle {selected_cycle} ({life_pct}% life)')
ax2.axhline(y=0.5, color='orange', linestyle='--', label='Risk Threshold (50%)')
ax2.fill_between(engine_data['cycle'].values, probs, alpha=0.2, color='crimson')
ax2.set_xlabel('Cycle')
ax2.set_ylabel('Failure Probability')
ax2.set_ylim(0, 1)
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

# ── Feature Importance ────────────────────────────────────────────────
st.subheader("🔍 Key Failure Indicators")
importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
fig3, ax3 = plt.subplots(figsize=(7, 4))
importance.plot(kind='barh', ax=ax3, color='steelblue')
ax3.set_title('Sensor Importance Score')
plt.tight_layout()
st.pyplot(fig3)

# ── Anomaly Detection ─────────────────────────────────────────────────
st.divider()
st.subheader("🔬 Sensor Anomaly Detection")
st.caption(
    f"Each sensor on EQ-{engine_id:03d} is compared against its normal baseline "
    f"(learned from healthy equipment with RUL > 60). "
    f"A sensor is flagged anomalous if its current reading falls outside the normal range."
)

# Get current sensor readings at selected life %
current_readings = engine_data[sensor_cols].iloc[row_idx]

# Run each sensor through its Isolation Forest
anomaly_results = []
for sensor in sensor_cols:
    value = float(current_readings[sensor])
    prediction = detectors[sensor].predict([[value]])[0]  # 1=normal, -1=anomalous
    score = detectors[sensor].decision_function([[value]])[0]  # lower = more anomalous

    # Get normal range from healthy data for this sensor
    healthy_vals = train[train['RUL'] > 60][sensor]
    normal_min = round(float(healthy_vals.quantile(0.05)), 3)
    normal_max = round(float(healthy_vals.quantile(0.95)), 3)

    is_anomaly = prediction == -1
    anomaly_results.append({
        'Sensor'        : sensor.upper(),
        'Current Value' : round(value, 3),
        'Normal Min'    : normal_min,
        'Normal Max'    : normal_max,
        'Status'        : "🔴 Anomalous" if is_anomaly else "🟢 Normal",
        'Anomaly Score' : round(score, 4)  # more negative = more anomalous
    })

anomaly_df = pd.DataFrame(anomaly_results)
anomalous = anomaly_df[anomaly_df['Status'] == '🔴 Anomalous']
normal_sensors = anomaly_df[anomaly_df['Status'] == '🟢 Normal']

# Summary
a1, a2 = st.columns(2)
a1.metric("🔴 Anomalous Sensors", len(anomalous))
a2.metric("🟢 Normal Sensors", len(normal_sensors))

# Alert if any anomalies found
if len(anomalous) > 0:
    sensor_list = ', '.join(anomalous['Sensor'].tolist())
    if risk_prob <= 0.4:
        st.warning(
            f"⚠️ EARLY WARNING: Overall risk score is low but "
            f"{len(anomalous)} sensor(s) are behaving abnormally: **{sensor_list}**. "
            f"Monitor closely — this may indicate early-stage degradation."
        )
    else:
        st.error(
            f"🚨 {len(anomalous)} sensor(s) showing abnormal readings: **{sensor_list}**. "
            f"Combined with high risk score — immediate inspection recommended."
        )
else:
    st.success(f"✅ All sensors on EQ-{engine_id:03d} are reading within normal range.")

# Anomaly table sorted — anomalous first
anomaly_df_sorted = anomaly_df.sort_values('Anomaly Score', ascending=True)
st.dataframe(anomaly_df_sorted, use_container_width=True, hide_index=True)

# Anomaly trend chart — show anomaly score over full lifecycle
st.markdown("##### Anomaly Score Over Time (lower = more abnormal)")
st.caption("Shows how each sensor's anomaly score evolves over the equipment's lifecycle. "
           "A dropping score means the sensor is drifting away from normal behaviour.")

fig4, ax4 = plt.subplots(figsize=(12, 4))
for sensor in sensor_cols:
    scores = [
        detectors[sensor].decision_function([[v]])[0]
        for v in engine_data[sensor].values
    ]
    ax4.plot(engine_data['cycle'].values, scores, linewidth=1, alpha=0.7, label=sensor.upper())

ax4.axvline(x=selected_cycle, color='black', linestyle='--', linewidth=2,
            label=f'Current cycle {selected_cycle}')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Anomaly threshold')
ax4.set_xlabel('Cycle')
ax4.set_ylabel('Anomaly Score (negative = anomalous)')
ax4.legend(fontsize=7, ncol=3)
plt.tight_layout()
st.pyplot(fig4)

# ── Fleet Overview Table (synced with slider) ─────────────────────────
st.divider()
st.subheader("🏭 Fleet-wide Equipment Risk Overview")
st.caption(
    f"All 100 equipments evaluated at **{life_pct}% of their own lifecycle**. "
    f"Drag the slider to explore how risk evolves across the fleet over time."
)

fleet_data = []
for eid in sorted(train['engine_id'].unique()):
    edata = train[train['engine_id'] == eid].copy().reset_index(drop=True)
    n = len(edata)
    midx = min(int(n * life_pct / 100), n - 1)
    prob = model.predict_proba(
        scaler.transform(edata[feature_cols].iloc[[midx]])
    )[0][1]
    rul_val = int(edata.iloc[midx]['RUL'])
    cycle_val = int(edata.iloc[midx]['cycle'])
    total_life = int(edata['cycle'].max())

    if prob > 0.7:
        status = "🔴 Critical"
    elif prob > 0.4:
        status = "🟡 Warning"
    else:
        status = "🟢 Normal"

    fleet_data.append({
        'Equipment ID': f'EQ-{eid:03d}',
        'Total Life (cycles)': total_life,
        'Cycle at snapshot': cycle_val,
        'RUL': rul_val,
        'Failure Risk %': round(prob * 100, 1),
        'Status': status
    })

fleet_df = pd.DataFrame(fleet_data)

col1, col2, col3 = st.columns(3)
critical = len(fleet_df[fleet_df['Status'] == '🔴 Critical'])
warning  = len(fleet_df[fleet_df['Status'] == '🟡 Warning'])
normal   = len(fleet_df[fleet_df['Status'] == '🟢 Normal'])
col1.metric("🔴 Critical", critical)
col2.metric("🟡 Warning", warning)
col3.metric("🟢 Normal", normal)

fleet_df_sorted = fleet_df.sort_values('Failure Risk %', ascending=False)
st.dataframe(fleet_df_sorted, use_container_width=True, hide_index=True)

# ── Maintenance Schedule ──────────────────────────────────────────────
st.divider()
st.subheader("🔧 Maintenance Schedule Recommender")
st.caption(
    "Each equipment is snapshotted at the point where **RUL ≥ 60** (well before the danger zone). "
    "The model then scans forward to find how many cycles until failure risk crosses 70%, "
    "and assigns a maintenance priority accordingly."
)

@st.cache_data
def build_maintenance_schedule(_model, _scaler, train, feature_cols):
    schedule = []
    for eid in sorted(train['engine_id'].unique()):
        edata = train[train['engine_id'] == eid].copy().reset_index(drop=True)
        n = len(edata)

        healthy_rows = edata[edata['RUL'] >= 60]
        if len(healthy_rows) > 0:
            latest_idx = healthy_rows.index[-1]
        else:
            latest_idx = 0

        latest_row    = edata.iloc[latest_idx]
        current_cycle = int(latest_row['cycle'])
        current_rul   = int(latest_row['RUL'])

        current_risk = _model.predict_proba(
            _scaler.transform(edata[feature_cols].iloc[[latest_idx]])
        )[0][1]

        if current_risk > 0.7:
            cycles_to_critical = 0
        else:
            cycles_to_critical = None
            for future_idx in range(latest_idx, n):
                future_risk = _model.predict_proba(
                    _scaler.transform(edata[feature_cols].iloc[[future_idx]])
                )[0][1]
                if future_risk > 0.7:
                    cycles_to_critical = int(edata.iloc[future_idx]['cycle']) - current_cycle
                    break
            if cycles_to_critical is None:
                cycles_to_critical = current_rul

        if cycles_to_critical == 0:
            priority = "🔴 Immediate"
            action   = "Shut down and inspect now"
        elif cycles_to_critical <= 10:
            priority = "🟠 This Week"
            action   = "Schedule maintenance within 7 days"
        elif cycles_to_critical <= 30:
            priority = "🟡 This Month"
            action   = "Plan maintenance within 30 days"
        else:
            priority = "🟢 Monitor"
            action   = "No action needed, continue monitoring"

        schedule.append({
            'Equipment ID'       : f'EQ-{eid:03d}',
            'Current Risk %'     : round(current_risk * 100, 1),
            'Current RUL'        : current_rul,
            'Cycles to Critical' : cycles_to_critical,
            'Priority'           : priority,
            'Recommended Action' : action
        })

    return pd.DataFrame(schedule)

with st.spinner("Calculating maintenance schedule for all 100 equipments..."):
    schedule_df = build_maintenance_schedule(model, scaler, train, feature_cols)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 Immediate",  len(schedule_df[schedule_df['Priority'] == '🔴 Immediate']))
c2.metric("🟠 This Week",  len(schedule_df[schedule_df['Priority'] == '🟠 This Week']))
c3.metric("🟡 This Month", len(schedule_df[schedule_df['Priority'] == '🟡 This Month']))
c4.metric("🟢 Monitor",    len(schedule_df[schedule_df['Priority'] == '🟢 Monitor']))

immediate = schedule_df[schedule_df['Priority'] == '🔴 Immediate']
if not immediate.empty:
    st.error(f"🚨 {len(immediate)} equipment(s) require IMMEDIATE shutdown and inspection! See table below.")

priority_order = {'🔴 Immediate': 0, '🟠 This Week': 1, '🟡 This Month': 2, '🟢 Monitor': 3}
schedule_df['_order'] = schedule_df['Priority'].map(priority_order)
schedule_df_sorted = schedule_df.drop(columns=['_order']).iloc[
    schedule_df['_order'].argsort()
]

st.dataframe(schedule_df_sorted, use_container_width=True, hide_index=True)

csv = schedule_df_sorted.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Maintenance Schedule as CSV",
    data=csv,
    file_name="ongc_maintenance_schedule.csv",
    mime="text/csv"
)

st.divider()
st.caption("ONGC Predictive Maintenance System | Built with XGBoost + Streamlit | AUC: 0.9911")