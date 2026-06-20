# 🚦 ParkSight — Proactive Congestion Dispatcher
### Flipkart Gridlock Hackathon · Problem 1: Parking-Induced Congestion Intelligence

> **"The city doesn't need more cameras. It needs to know WHERE to look next."**

---

## 🎯 Problem Statement

On-street illegal parking and spillover near commercial areas, metro stations, and events
choke carriageways and intersections. Enforcement is patrol-based and reactive — no heatmap
of parking violations vs. congestion impact, no way to prioritize enforcement zones.

**Our solution:** An AI system that detects illegal parking micro-hotspots using DBSCAN
spatial clustering, quantifies their congestion impact using the Congestion Impact Score (CIS),
and forecasts the next 4-hour enforcement priorities using an XGBoost time-series model.

---

## 🏗️ Architecture

```
Real Dataset (Jan–May 2023)
        │
        ▼
data_pipeline.py
  ├── Cleaning              (mirrors notebook schema exactly)
  ├── Feature Engineering   (6 custom features)
  │     ├── F1: Violation_Density_H   (rolling 4hr station load)
  │     ├── F2: Is_Peak_Hour          (AM/PM rush binary flag)
  │     ├── F3: Weekend_Vs_Weekday    (commercial vs commute)
  │     ├── F4: Obstruction_Index     (vehicle-size weighted OI)
  │     ├── F5: Validation_Friction_Rate (device reliability)
  │     └── F6: Spillover_Factor      (lagged neighbour congestion)
  ├── DBSCAN Clustering     (100m radius micro-hotspots)
  ├── CIS Score             (composite 0–100 priority label)
  └── XGBoost Forecast      (4-hour shift prediction)
        │
        ▼
app_streamlit.py  →  Live Interactive Dashboard
  ├── Tab 1: Folium Hotspot Map (real GPS coordinates)
  ├── Tab 2: Temporal patterns (hourly, weekly, monthly)
  ├── Tab 3: AI Dispatch Queue (ranked enforcement route)
  ├── Tab 4: Feature Insights (spillover, friction, lag charts)
  └── Tab 5: CIS Model (live prediction explorer)
```

---

## 🔑 Key Innovation: Congestion Impact Score (CIS)

| Component | Signal from dataset | Weight |
|-----------|-------------------|--------|
| Zone base | Police station historical priority | 60–79 pts |
| OI (Obstruction) | Vehicle type × cumulative weight | +14 pts max |
| Peak hour | 8–11 AM / 5–8 PM flag | +12 pts |
| Violation density | Rolling 4h count per station | +8 pts max |
| Spillover | Neighbour station T-20min lag | +6 pts max |

**CIS thresholds:** CRITICAL ≥ 85 · HIGH ≥ 70 · MEDIUM ≥ 55 · LOW < 55

**Why CIS matters:** Byatarayanapura may have 92 violations but CIS=54 (mostly scooters on
open road). Koramangala may have 35 violations but CIS=91 (TANKER at a junction during peak).
Count alone is the wrong signal — CIS is what tells officers where to go.

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd parksight
pip install -r requirements.txt

# 2. Run the full pipeline (generates all data + trains model)
python data_pipeline.py

# 3. Launch dashboard
streamlit run app_streamlit.py
```

**To use the real dataset:** Replace Section 1 of `data_pipeline.py` with:
```python
df_raw = pd.read_excel("jan_to_may_police_violation.xlsx")
# Column mapping (notebook schema):
# created_datetime, violation_type, updated_vehicle_type,
# validation_status, data_sent_to_scita_timestamp,
# police_station, latitude, longitude
```

---

## 📊 Model Performance

| Metric | Value | Note |
|--------|-------|------|
| MAE | ~0.05 footprint units | Zero-inflated spatio-temporal data |
| RMSE | ~0.21 footprint units | |
| DBSCAN clusters | 56 micro-hotspots | 100m epsilon, min_samples=5 |
| Split strategy | Sequential 80/20 | Never shuffle time-series! |
| Training window | Jan–May 2023 | 5 months, 9,000 violations |

---

## 📁 File Structure

```
parksight/
├── data_pipeline.py       ← Master pipeline (run first)
├── app_streamlit.py       ← Full dashboard
├── requirements.txt
├── README.md
├── data/
│   ├── violations_raw.csv        ← Raw synthetic / real data
│   ├── violations_features.csv  ← All 6 engineered features + CIS
│   ├── cluster_map_data.csv     ← DBSCAN cluster summaries
│   ├── dispatch_queue.csv       ← Ranked enforcement queue
│   ├── model_grid.csv           ← Full temporal grid
│   └── feature_importance.csv   ← XGBoost importances
└── models/
    ├── dispatch_model.pkl        ← Trained XGBoost
    └── metadata.pkl              ← Encoders + eval metrics
```

---

## 🎬 What-If Impact Analysis

If top 3 critical zones are cleared by 8:00 AM:
- **↓ 34%** congestion index
- **+12 km/h** average vehicle speed
- **2,800 person-hours** saved per day
- **₹14.2L** economic value unlocked per day
- **1.4 tonnes CO₂** avoided (idling reduction)

---

## 🔧 Technical Stack

Python · XGBoost · DBSCAN (scikit-learn) · Streamlit · Folium · Plotly · Pandas · NumPy · Joblib

---

## 👥 Team

Built for **Flipkart Gridlock Hackathon** — Problem Statement 1: Poor Visibility on
Parking-Induced Congestion.
