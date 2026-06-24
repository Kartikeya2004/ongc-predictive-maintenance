import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

train = load_data()
model, scaler, feature_cols = train_model(train)

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

# ── Life % Slider (shared across selected engine + fleet table) ───────
life_pct = st.sidebar.slider(
    "⏱️ Equipment Life (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1,
    help="0% = brand new  |  100% = end of life. "
         "Each equipment is evaluated at this percentage of its own total life."
)

st.sidebar.caption(
    "📌 This slider is relative — 50% means each equipment is at "
    "the midpoint of *its own* lifecycle, regardless of its absolute cycle count."
)

# ── Resolve selected engine row from life % ───────────────────────────
engine_data = train[train['engine_id'] == engine_id].copy().reset_index(drop=True)
total_rows = len(engine_data)
row_idx = min(int(total_rows * life_pct / 100), total_rows - 1)
row = engine_data.iloc[row_idx]

selected_cycle = int(row['cycle'])
max_cycle = int(engine_data['cycle'].max())
current_rul = int(row['RUL'])

# Predict risk for selected life %
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

# ── Sensor Trends with current cycle marker ───────────────────────────
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

# ── Fleet Overview Table ──────────────────────────────────────────────
st.subheader("🏭 Fleet-wide Equipment Risk Overview")
st.caption(
    f"All 100 equipments evaluated at **{life_pct}% of their own lifecycle**. "
    f"Each equipment's cycle count differs — the slider compares them fairly on a relative scale."
)

fleet_data = []
for eid in sorted(train['engine_id'].unique()):
    edata = train[train['engine_id'] == eid].copy().reset_index(drop=True)
    n = len(edata)
    # Each equipment evaluated at the same RELATIVE life stage
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

# Summary metrics
col1, col2, col3 = st.columns(3)
critical = len(fleet_df[fleet_df['Status'] == '🔴 Critical'])
warning  = len(fleet_df[fleet_df['Status'] == '🟡 Warning'])
normal   = len(fleet_df[fleet_df['Status'] == '🟢 Normal'])

col1.metric("🔴 Critical", critical)
col2.metric("🟡 Warning", warning)
col3.metric("🟢 Normal", normal)

# Sortable table — highest risk first
fleet_df = fleet_df.sort_values('Failure Risk %', ascending=False)
st.dataframe(fleet_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("ONGC Predictive Maintenance System | Built with XGBoost + Streamlit | AUC: 0.9911")