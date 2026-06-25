import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta

plt.style.use('dark_background')
import matplotlib as mpl
mpl.rcParams['axes.facecolor'] = '#1a1a2e'
mpl.rcParams['figure.facecolor'] = '#1a1a2e'
mpl.rcParams['grid.color'] = '#333355'
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'

st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

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
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=17531/3100, random_state=42, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    return model, scaler, feature_cols

@st.cache_resource
def train_anomaly_detectors(train, feature_cols):
    healthy = train[train['RUL'] > 60]
    detectors = {}
    for sensor in feature_cols:
        if sensor == 'cycle':
            continue
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(healthy[[sensor]])
        detectors[sensor] = iso
    return detectors

@st.cache_data
def build_maintenance_schedule(_model, _scaler, train, feature_cols):
    schedule = []
    for eid in sorted(train['engine_id'].unique()):
        edata = train[train['engine_id'] == eid].copy().reset_index(drop=True)
        n = len(edata)
        healthy_rows = edata[edata['RUL'] >= 60]
        latest_idx = healthy_rows.index[-1] if len(healthy_rows) > 0 else 0
        latest_row = edata.iloc[latest_idx]
        current_cycle = int(latest_row['cycle'])
        current_rul = int(latest_row['RUL'])
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
            action = "Shut down and inspect now"
        elif cycles_to_critical <= 10:
            priority = "🟠 This Week"
            action = "Schedule maintenance within 7 days"
        elif cycles_to_critical <= 30:
            priority = "🟡 This Month"
            action = "Plan maintenance within 30 days"
        else:
            priority = "🟢 Monitor"
            action = "No action needed, continue monitoring"
        schedule.append({
            'Equipment ID': f'EQ-{eid:03d}',
            'Current Risk %': round(current_risk * 100, 1),
            'Current RUL': current_rul,
            'Cycles to Critical': cycles_to_critical,
            'Priority': priority,
            'Recommended Action': action
        })
    return pd.DataFrame(schedule)

# ── Load everything ───────────────────────────────────────────────────
train = load_data()
model, scaler, feature_cols = train_model(train)
sensor_cols = [c for c in feature_cols if c != 'cycle']
detectors = train_anomaly_detectors(train, feature_cols)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Industrial Predictive Maintenance System")
st.sidebar.divider()

engine_id = st.sidebar.selectbox(
    "Select Equipment ID",
    sorted(train['engine_id'].unique()),
    format_func=lambda x: f"EQ-{x:03d}"
)

life_pct = st.sidebar.slider(
    "Equipment Life (%)",
    min_value=0, max_value=100, value=50, step=1,
    help="0% = brand new | 100% = end of life"
)

# ── Resolve selected engine ───────────────────────────────────────────
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

# ── Header ────────────────────────────────────────────────────────────
st.title("🛢️ Industrial Equipment Health Monitor")
st.markdown("AI-powered real-time failure risk prediction using Machine Learning")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📊 Sensor Analysis",
    "🏭 Fleet Monitor",
    "🔧 Maintenance",
    "💰 Cost Analysis"
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equipment ID", f"EQ-{engine_id:03d}")
    col2.metric("Life Stage", f"{life_pct}%")
    col3.metric("Actual Cycle", selected_cycle)
    col4.metric("Remaining Useful Life", f"{current_rul} cycles")
    col5.metric("Failure Risk", f"{risk_prob*100:.1f}%",
                delta="⚠️ High Risk" if risk_prob > 0.5 else "✅ Normal",
                delta_color="inverse")

    st.divider()

    if risk_prob > 0.7:
        st.error(f"🚨 CRITICAL: Equipment EQ-{engine_id:03d} has {risk_prob*100:.1f}% failure probability. Immediate maintenance required!")
    elif risk_prob > 0.4:
        st.warning(f"⚠️ WARNING: Equipment EQ-{engine_id:03d} showing early failure signs. Schedule maintenance soon.")
    else:
        st.success(f"✅ Equipment EQ-{engine_id:03d} is operating normally at {life_pct}% of its life (cycle {selected_cycle}).")

    st.divider()

    st.subheader("📈 Failure Risk Over Time")
    probs = model.predict_proba(scaler.transform(engine_data[feature_cols]))[:, 1]
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(engine_data['cycle'].values, probs, color='crimson', linewidth=1.5)
    ax2.axvline(x=selected_cycle, color='dodgerblue', linestyle='--', linewidth=2,
                label=f'Current: Cycle {selected_cycle} ({life_pct}% life)')
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='Risk Threshold (50%)')
    ax2.fill_between(engine_data['cycle'].values, probs, alpha=0.2, color='crimson')
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Failure Probability')
    ax2.set_ylim(0, 1)
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()

    st.subheader("🔍 Key Failure Indicators")
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    colors = ['crimson' if imp == importance.max() else 'steelblue' for imp in importance]
    importance.plot(kind='barh', ax=ax3, color=colors)
    ax3.set_title('Sensor Importance Score')
    ax3.set_xlabel('Importance')
    plt.tight_layout()
    st.pyplot(fig3)

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — SENSOR ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"📡 Sensor Readings — EQ-{engine_id:03d} at {life_pct}% life")

    top_sensors = ['sensor11', 'sensor4', 'sensor9']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, sensor in zip(axes, top_sensors):
        ax.plot(engine_data['cycle'], engine_data[sensor], color='steelblue', linewidth=1.2)
        ax.axvline(x=selected_cycle, color='red', linestyle='--', label=f'Cycle {selected_cycle}')
        ax.set_title(f'{sensor.upper()} Reading')
        ax.set_xlabel('Cycle')
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()

    st.subheader("🔬 Sensor Anomaly Detection")
    st.caption("Sensors compared against healthy baseline (RUL > 60). Flagged if outside normal range.")

    current_readings = engine_data[sensor_cols].iloc[row_idx]
    anomaly_results = []
    for sensor in sensor_cols:
        value = float(current_readings[sensor])
        prediction = detectors[sensor].predict([[value]])[0]
        score = detectors[sensor].decision_function([[value]])[0]
        healthy_vals = train[train['RUL'] > 60][sensor]
        normal_min = round(float(healthy_vals.quantile(0.05)), 3)
        normal_max = round(float(healthy_vals.quantile(0.95)), 3)
        is_anomaly = prediction == -1
        anomaly_results.append({
            'Sensor': sensor.upper(),
            'Current Value': round(value, 3),
            'Normal Min': normal_min,
            'Normal Max': normal_max,
            'Status': "🔴 Anomalous" if is_anomaly else "🟢 Normal",
            'Anomaly Score': round(score, 4)
        })

    anomaly_df = pd.DataFrame(anomaly_results)
    anomalous = anomaly_df[anomaly_df['Status'] == '🔴 Anomalous']
    normal_sensors = anomaly_df[anomaly_df['Status'] == '🟢 Normal']

    a1, a2 = st.columns(2)
    a1.metric("🔴 Anomalous Sensors", len(anomalous))
    a2.metric("🟢 Normal Sensors", len(normal_sensors))

    if len(anomalous) > 0:
        sensor_list = ', '.join(anomalous['Sensor'].tolist())
        if risk_prob <= 0.4:
            st.warning(f"⚠️ EARLY WARNING: {len(anomalous)} sensor(s) behaving abnormally: **{sensor_list}**. Monitor closely.")
        else:
            st.error(f"🚨 {len(anomalous)} sensor(s) anomalous: **{sensor_list}**. Immediate inspection recommended.")
    else:
        st.success(f"✅ All sensors on EQ-{engine_id:03d} reading within normal range.")

    anomaly_df_sorted = anomaly_df.sort_values('Anomaly Score', ascending=True)
    st.dataframe(anomaly_df_sorted, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — FLEET MONITOR
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"🏭 All Equipment at {life_pct}% Lifecycle")
    st.caption("Drag the Equipment Life slider to see how risk evolves across the fleet.")

    fleet_data = []
    for eid in sorted(train['engine_id'].unique()):
        edata = train[train['engine_id'] == eid].copy().reset_index(drop=True)
        n = len(edata)
        midx = min(int(n * life_pct / 100), n - 1)
        prob = model.predict_proba(scaler.transform(edata[feature_cols].iloc[[midx]]))[0][1]
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
            'Cycle': cycle_val,
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

    st.dataframe(fleet_df.sort_values('Failure Risk %', ascending=False),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — MAINTENANCE
# ══════════════════════════════════════════════════════════════════════
with tab4:
    with st.spinner("Calculating maintenance schedule..."):
        schedule_df = build_maintenance_schedule(model, scaler, train, feature_cols)

    st.subheader("🔧 Maintenance Schedule — All Equipment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Immediate",  len(schedule_df[schedule_df['Priority'] == '🔴 Immediate']))
    c2.metric("🟠 This Week",  len(schedule_df[schedule_df['Priority'] == '🟠 This Week']))
    c3.metric("🟡 This Month", len(schedule_df[schedule_df['Priority'] == '🟡 This Month']))
    c4.metric("🟢 Monitor",    len(schedule_df[schedule_df['Priority'] == '🟢 Monitor']))

    immediate = schedule_df[schedule_df['Priority'] == '🔴 Immediate']
    if not immediate.empty:
        st.error(f"🚨 {len(immediate)} equipment(s) require IMMEDIATE shutdown and inspection!")

    priority_order = {'🔴 Immediate': 0, '🟠 This Week': 1, '🟡 This Month': 2, '🟢 Monitor': 3}
    schedule_df['_order'] = schedule_df['Priority'].map(priority_order)
    schedule_sorted = schedule_df.drop(columns=['_order']).iloc[schedule_df['_order'].argsort()]
    st.dataframe(schedule_sorted, use_container_width=True, hide_index=True)

    csv = schedule_sorted.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download as CSV", data=csv,
                       file_name="maintenance_schedule.csv", mime="text/csv")

    st.divider()

    st.subheader(f"⏱️ Downtime Estimator — EQ-{engine_id:03d}")
    engine_schedule = schedule_df[schedule_df['Equipment ID'] == f'EQ-{engine_id:03d}'].iloc[0]
    cycles_to_critical = int(engine_schedule['Cycles to Critical'])

    col1, col2 = st.columns(2)
    with col1:
        cycle_duration_hours = st.number_input("Duration of one cycle (hours)", min_value=1, max_value=48, value=8, step=1)
    with col2:
        repair_duration_hours = st.number_input("Repair duration (hours)", min_value=1, max_value=240, value=24, step=1)

    hours_to_maintenance = cycles_to_critical * cycle_duration_hours
    days_to_maintenance = hours_to_maintenance / 24
    maintenance_date = datetime.now() + timedelta(hours=hours_to_maintenance)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cycles to Critical", cycles_to_critical)
    m2.metric("Days Until Maintenance", f"{days_to_maintenance:.1f}")
    m3.metric("Maintenance Date", maintenance_date.strftime("%d %b %Y"))
    m4.metric("Expected Downtime", f"{repair_duration_hours} hrs")

    fig_time, ax_time = plt.subplots(figsize=(12, 2))
    back_online_x = days_to_maintenance + repair_duration_hours / 24
    ax_time.barh(0, days_to_maintenance, color='steelblue', height=0.4, label='Safe window')
    ax_time.barh(0, repair_duration_hours/24, left=days_to_maintenance, color='orange', height=0.4, label='Maintenance')
    ax_time.barh(0, 10, left=back_online_x, color='green', height=0.4, label='Back online')
    ax_time.axvline(x=0, color='white', linewidth=1.5, linestyle='--')
    ax_time.axvline(x=days_to_maintenance, color='red', linewidth=1.5, linestyle='--')
    ax_time.set_xlabel('Days from now')
    ax_time.set_title(f'EQ-{engine_id:03d} — Maintenance Timeline')
    ax_time.set_yticks([])
    ax_time.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_time)

    st.divider()

    st.subheader("📉 Production Loss During Downtime")
    col1, col2 = st.columns(2)
    with col1:
        production_per_hour = st.number_input(
            "Production rate (barrels/hour)",
            min_value=10, max_value=10000, value=500, step=10
        )
    with col2:
        oil_price_per_barrel = st.number_input(
            "Oil price (₹ per barrel)",
            min_value=1000, max_value=20000, value=6500, step=100
        )

    lost_barrels = production_per_hour * repair_duration_hours
    lost_revenue = lost_barrels * oil_price_per_barrel / 100000

    c1, c2 = st.columns(2)
    c1.metric("Lost Production", f"{lost_barrels:,} barrels")
    c2.metric("Revenue Loss During Downtime", f"₹{lost_revenue:.1f} lakhs")

    if days_to_maintenance <= 10:
        st.error(f"🚨 EQ-{engine_id:03d} needs maintenance within {days_to_maintenance:.1f} days!")
    elif days_to_maintenance <= 30:
        st.warning(f"⚠️ Schedule maintenance by {maintenance_date.strftime('%d %b %Y')}.")
    else:
        st.success(f"✅ {days_to_maintenance:.1f} days until maintenance needed. Plan at your convenience.")

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — COST ANALYSIS
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("💰 Maintenance Cost Savings Estimator")

    col1, col2 = st.columns(2)
    with col1:
        reactive_cost = st.number_input("Cost of reactive failure (₹ lakhs)", min_value=1.0, max_value=500.0, value=50.0, step=1.0)
        predictive_cost = st.number_input("Cost of planned maintenance (₹ lakhs)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
    with col2:
        failures_per_year = st.number_input("Failures per year (without system)", min_value=1, max_value=100, value=20, step=1)
        catch_rate = st.slider("Model catch rate (%)", min_value=50, max_value=100, value=92)

    caught = int(failures_per_year * catch_rate / 100)
    missed = failures_per_year - caught
    reactive_total = failures_per_year * reactive_cost
    predictive_total = (caught * predictive_cost) + (missed * reactive_cost)
    savings = reactive_total - predictive_total

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Without AI (₹ lakhs/year)", f"₹{reactive_total:.1f}L")
    m2.metric("With AI System (₹ lakhs/year)", f"₹{predictive_total:.1f}L")
    m3.metric("Annual Savings", f"₹{savings:.1f}L",
              delta=f"↓ {round(savings/reactive_total*100)}% reduction", delta_color="normal")

    if savings > 0:
        st.success(f"✅ This system saves approximately ₹{savings:.1f} lakhs per year — a {round(savings/reactive_total*100)}% reduction in maintenance costs.")

    fig_roi, ax_roi = plt.subplots(figsize=(7, 4))
    bars = ax_roi.bar(['Reactive\n(No AI)', 'Predictive\n(With AI)'],
                      [reactive_total, predictive_total],
                      color=['crimson', 'steelblue'], width=0.4)
    ax_roi.set_ylabel('Annual Cost (₹ Lakhs)')
    ax_roi.set_title('Reactive vs Predictive Maintenance Cost')
    for bar, val in zip(bars, [reactive_total, predictive_total]):
        ax_roi.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'₹{val:.1f}L', ha='center', fontsize=11, fontweight='bold', color='white')
    plt.tight_layout()
    st.pyplot(fig_roi)

st.divider()
st.caption("Industrial Predictive Maintenance System | Built with XGBoost + Streamlit | AUC: 0.9911")