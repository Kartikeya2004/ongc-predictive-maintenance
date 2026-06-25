# Industrial Predictive Maintenance System

An ML-based equipment failure prediction system built during an internship at an oil & gas company. The system monitors industrial equipment sensor data and predicts failure risk in real time using Machine Learning — deployed as a fully interactive multi-tab Streamlit dashboard.

## Dashboard Features
- **🏠 Overview** — Equipment health metrics, risk score, and failure probability over time
- **📊 Sensor Analysis** — Per-sensor degradation trends and anomaly detection using Isolation Forest
- **🏭 Fleet Monitor** — Risk status of all 100 equipments synced with lifecycle slider
- **🔧 Maintenance** — Automated maintenance schedule with downtime and production loss estimator
- **💰 Cost Analysis** — Annual savings calculator comparing reactive vs predictive maintenance

## Project Highlights
- **AUC Score: 0.9911** — near perfect discrimination between safe and at-risk equipment
- **92% Recall** — catches 9 out of 10 real failures before they happen
- **Dual ML system** — XGBoost for risk prediction + Isolation Forest for anomaly detection
- **Business impact** — cost savings estimator shows ROI in lakhs per year
- Trained on NASA CMAPSS dataset (100 engines, 20,631 sensor readings)

## Tech Stack
- **ML Models:** XGBoost Classifier, Isolation Forest
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib
- **Dashboard:** Streamlit (multi-tab layout)
- **Dataset:** NASA CMAPSS (publicly available)

## How It Works
1. Sensor readings collected from industrial equipment every operational cycle
2. 13 low-variance sensors filtered out — only 8 meaningful sensors used
3. XGBoost predicts whether equipment will fail within the next 30 cycles
4. Isolation Forest flags individual sensors behaving abnormally (early warning)
5. Class imbalance (85% safe vs 15% at-risk) handled using scale_pos_weight
6. Maintenance schedule generated with exact dates, downtime, and production loss estimates

## Key Findings
- **Sensor 11** is the strongest failure predictor (62.7% feature importance)
- **Sensor 4** (11.3%) and **Sensor 9** (7.1%) are secondary indicators
- Equipment shows clear sensor degradation trend 30+ cycles before failure
- Anomaly detection catches early-stage degradation before risk score rises

## Results
| Metric | Value |
|--------|-------|
| AUC Score | 0.9911 |
| Accuracy | 95% |
| Recall (At Risk) | 92% |
| Precision (At Risk) | 81% |

## Setup and Installation
pip install pandas numpy scikit-learn xgboost matplotlib streamlit

Download the NASA CMAPSS dataset and place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in the project directory.

streamlit run dashboard.py

## Project Structure
predictive-maintenance-system/
├── dashboard.py        # Main Streamlit app (5 tabs)
├── requirements.txt    # Python dependencies
└── README.md

## Author
Kartikeya — B.Tech CSE, SRM Institute of Science and Technology
Internship: Oil & Gas Sector (Predictive Maintenance Project)
