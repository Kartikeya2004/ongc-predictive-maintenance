# ONGC Predictive Maintenance System

An ML-based equipment failure prediction system built during an internship at **Oil and Natural Gas Corporation (ONGC)**. The system monitors industrial equipment sensor data and predicts failure risk in real time using Machine Learning.

## Live Dashboard
Run locally using Streamlit — select any equipment ID and drag the cycle slider to see failure risk evolve over time.

## Project Highlights
- **AUC Score: 0.9911** — near perfect discrimination between safe and at-risk equipment
- **92% Recall** — catches 9 out of 10 real failures before they happen
- **Real-time dashboard** with interactive cycle slider built using Streamlit
- Trained on NASA CMAPSS turbofan engine degradation dataset (100 engines, 20,631 sensor readings)

## Tech Stack
- **ML Model:** XGBoost Classifier
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Dataset:** NASA CMAPSS (publicly available)

## How It Works
1. Sensor readings are collected from industrial equipment every operational cycle
2. 13 low-variance sensors are filtered out — only 8 meaningful sensors are used
3. XGBoost model predicts whether equipment will fail within the next 30 cycles
4. Class imbalance (85% safe vs 15% at-risk) handled using `scale_pos_weight`
5. Results displayed on a live Streamlit dashboard with risk alerts

## Key Findings
- **Sensor 11** is the strongest failure predictor (62.7% feature importance)
- **Sensor 4** and **Sensor 9** are secondary indicators
- Equipment shows clear sensor degradation trend 30+ cycles before failure

## Results
| Metric | Value |
|--------|-------|
| AUC Score | 0.9911 |
| Accuracy | 95% |
| Recall (At Risk) | 92% |
| Precision (At Risk) | 81% |

## Setup and Installation
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit
```

Download the NASA CMAPSS dataset and place `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in the project directory.

```bash
streamlit run dashboard.py
```

## Industry Application
While trained on NASA turbofan engine data, the same pipeline applies directly to ONGC oil pumps, compressors, and drilling motors — any industrial equipment with continuous sensor monitoring.

## Author
**Kartikeya** — B.Tech CSE, SRM Institute of Science and Technology  
Internship: Oil and Natural Gas Corporation (ONGC)
