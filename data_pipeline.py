"""
data_pipeline.py  —  ParkSight: Proactive Congestion Dispatcher
================================================================
Loads the REAL dataset from HackerEarth.

Real columns (confirmed from data snapshot):
  id, latitude, longitude, location, vehicle_number, vehicle_type,
  description, violation_type, offence_code, created_datetime,
  closed_datetime, modified_datetime, device_id, created_by_id,
  center_code, police_station, data_sent_to_scita, junction_name,
  action_taken_timestamp, data_sent_to_scita_timestamp,
  updated_vehicle_number, updated_vehicle_type,
  validation_status, validation_timestamp

NOTE on offence_code:
  Kept in the pipeline. Used to build a violation_code_map so every
  violation_type string is anchored to its real legal section code.
  e.g. "NO PARKING" → [113], "WRONG PARKING" → [112]

Run:
  python data_pipeline.py
  streamlit run app_streamlit.py
"""

import re
import os
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════
VEHICLE_MASS_MAP = {
    "SCOOTER":       1,
    "TWO WHEELER":   1,
    "BIKE":          1,
    "AUTO":          2,
    "AUTO-RICKSHAW": 2,
    "CAR":           3,
    "MAXI-CAB":      4,
    "BUS":           4,
    "TRUCK":         4,
    "TANKER":        5,
    "UNKNOWN":       2,
}

ADJACENCY = {
    "Madiwala":        "Bellandur",
    "Bellandur":       "Whitefield",
    "Whitefield":      "Indiranagar",
    "Indiranagar":     "Madiwala",
    "Byatarayanapura": "Upparpet",
    "Upparpet":        "Madiwala",
    "Shivajinagar":    "Rajajinagar",
    "Koramangala":     "Madiwala",
    "HSR Layout":      "Koramangala",
    "Rajajinagar":     "Yeshwanthpur",
    "Yeshwanthpur":    "Byatarayanapura",
    "Begur":           "HSR Layout",
}

FEATURES = [
    "hour", "day_of_week", "Is_Weekend", "month",
    "Lag_1_Shift", "Lag_1_Day", "Lag_1_Week",
]

# ════════════════════════════════════════════════════════════════
# SECTION 1 — Load real CSV
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("ParkSight — Data Pipeline  (Real Dataset)")
print("=" * 60)

FILE = "jan to may police violation_anonymized791b166.xlsx"
print(f"\n📂  Loading: {FILE}")
df_raw = pd.read_excel(FILE)
print(f"✅  {len(df_raw):,} rows × {df_raw.shape[1]} columns")
print(f"    Columns: {df_raw.columns.tolist()}")

# ════════════════════════════════════════════════════════════════
# SECTION 2 — Cleaning
# ════════════════════════════════════════════════════════════════
print("\n⚙   Cleaning …")

df = df_raw.copy()

# ── Columns we intentionally KEEP and why ────────────────────
# offence_code  → legal section number; used to build violation_code_map
#                 so dashboard can show which law section was violated
# device_id     → used for Validation Friction Rate
# center_code   → zone identifier kept for grouping
# junction_name → drives junction_known feature (+6 pts in CIS)
# data_sent_to_scita → enforcement gap KPI
# updated_vehicle_type → preferred over vehicle_type when available

# Columns with zero modelling value — drop these only
DROP = [
    "location",              # free-text address string
    "vehicle_number",        # anonymised, no signal
    "updated_vehicle_number",# anonymised, no signal
    "closed_datetime",       # almost entirely NULL
    "modified_datetime",     # admin timestamp, not a traffic signal
    "action_taken_timestamp",# almost entirely NULL
    "description",           # NULL in every row of real data
    "created_by_id",         # officer ID, not a traffic signal
]
df.drop(columns=[c for c in DROP if c in df.columns], inplace=True)

# ── Parse violation_type JSON array → primary label + codes ──
# Real format: ["WRONG PARKING","PARKING NEAR ROAD CROSSING"]
# offence_code format: [112,104]  or  [113]
def extract_primary_violation(raw):
    if pd.isna(raw):
        return "UNKNOWN"
    matches = re.findall(r'"([^"]+)"', str(raw))
    return matches[0].strip().upper() if matches else str(raw).strip().upper()

def extract_all_violations(raw):
    if pd.isna(raw):
        return []
    return [m.strip().upper() for m in re.findall(r'"([^"]+)"', str(raw))]

def extract_offence_codes(raw):
    """Return list of ints from '[112,104]' or '[113]'"""
    if pd.isna(raw):
        return []
    return [int(x) for x in re.findall(r'\d+', str(raw))]

df["violation_type"]   = df["violation_type"].apply(extract_primary_violation)
df["all_violations"]   = df["violation_type_raw"] if "violation_type_raw" in df.columns \
                          else df_raw["violation_type"].apply(extract_all_violations)
df["offence_codes"]    = df_raw["offence_code"].apply(extract_offence_codes)

# ── Build violation → offence_code lookup table ───────────────
# This is the mapping judges will care about: which law section = which violation
rows_map = []
for _, row in df.iterrows():
    vtype = row["violation_type"]
    codes = row["offence_codes"]
    for code in codes:
        rows_map.append({"violation_type": vtype, "offence_code": code})

violation_code_map = (
    pd.DataFrame(rows_map)
    .drop_duplicates()
    .sort_values(["violation_type", "offence_code"])
    .reset_index(drop=True)
)
violation_code_map.to_csv("data/violation_code_map.csv", index=False)
print(f"✅  Violation → offence code map:")
print(violation_code_map.to_string(index=False))

# ── Datetime ──────────────────────────────────────────────────
df["created_datetime"] = (
    pd.to_datetime(df["created_datetime"], format='ISO8601', utc=True)
    .dt.tz_localize(None)
)
df = df.sort_values("created_datetime").reset_index(drop=True)

# ── Vehicle type ──────────────────────────────────────────────
df["updated_vehicle_type"] = (
    df["updated_vehicle_type"]
    .fillna(df["vehicle_type"])
    .fillna("UNKNOWN")
    .astype(str).str.strip().str.upper()
)

# ── Other fields ──────────────────────────────────────────────
df["validation_status"] = (
    df["validation_status"].fillna("pending").astype(str).str.lower().str.strip()
)
df["is_sent_to_scita"] = (
    df["data_sent_to_scita"].astype(str).str.upper().eq("TRUE").astype(int)
)
df["police_station"] = df["police_station"].astype(str).str.strip()
df["junction_name"]  = df["junction_name"].fillna("No Junction").astype(str).str.strip()
df["junction_known"] = (df["junction_name"] != "No Junction").astype(int)

df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

print(f"\n    Shape after cleaning  : {df.shape}")
print(f"    Police stations       : {sorted(df['police_station'].astype(str).unique())}")
print(f"    Date range            : {df['created_datetime'].min()}  →  {df['created_datetime'].max()}")
print(f"    Top violation types   :\n{df['violation_type'].value_counts().head(8).to_string()}")
print(f"    Top vehicle types     :\n{df['updated_vehicle_type'].value_counts().head(8).to_string()}")

df.to_csv("data/violations_raw.csv", index=False)

# ════════════════════════════════════════════════════════════════
# SECTION 3 — Feature Engineering
# ════════════════════════════════════════════════════════════════
print("\n⚙   Feature engineering …")

df["hour"]        = df["created_datetime"].dt.hour
df["day_of_week"] = df["created_datetime"].dt.dayofweek
df["month"]       = df["created_datetime"].dt.month

# F2 — Is_Peak_Hour
df["Is_Peak_Hour"] = (
    ((df["hour"] >= 8)  & (df["hour"] <= 11)) |
    ((df["hour"] >= 17) & (df["hour"] <= 20))
).astype(int)

# F3 — Weekend_Vs_Weekday
df["Weekend_Vs_Weekday"] = (df["day_of_week"] >= 5).astype(int)

# F4 — Vehicle_Weight
df["Vehicle_Weight"] = df["updated_vehicle_type"].map(VEHICLE_MASS_MAP).fillna(2)

# F1 — Rolling 4h Violation Density + Obstruction Index
df = df.set_index("created_datetime").sort_index()

df["Violation_Density_H"] = (
    df.groupby("police_station")["violation_type"]
    .transform(lambda g: g.rolling("4h").count())
)
df["Obstruction_Index"] = (
    df.groupby("police_station")["Vehicle_Weight"]
    .transform(lambda g: g.rolling("4h").sum())
)

# F6 — Spillover Factor
df["time_floor_4h"] = df.index.floor("4h")
shift_grid = (
    df.groupby(["time_floor_4h", "police_station"])["Obstruction_Index"]
    .mean().unstack(fill_value=0)
)
lagged_values = []
for ts, row in df.iterrows():
    lag_t     = row["time_floor_4h"] - pd.Timedelta(hours=4)
    neighbour = ADJACENCY.get(row["police_station"], row["police_station"])
    try:    val = float(shift_grid.loc[lag_t, neighbour])
    except: val = 0.0
    lagged_values.append(val)
df["Spillover_Factor"] = lagged_values
df = df.reset_index()

# F5 — Validation Friction Rate per device_id
device_friction = (
    df.groupby("device_id")
    .apply(lambda g: (g["validation_status"] == "approved").sum() / max(len(g), 1))
    .rename("Validation_Friction_Rate").reset_index()
)
df = df.merge(device_friction, on="device_id", how="left")

# Physical_Footprint (same as Vehicle_Weight — used in model grid)
df["Physical_Footprint"] = df["Vehicle_Weight"].copy()

# CIS score — zone base derived from real data density
station_share = df["police_station"].value_counts(normalize=True)
ZONE_BASE = {st: min(round(50 + station_share.get(st, 0.04) * 500, 1), 85)
             for st in df["police_station"].unique()}

def norm(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

row_adjustment = (
    norm(df["Obstruction_Index"])   * 14 +
    df["Is_Peak_Hour"]              * 12 +
    norm(df["Violation_Density_H"]) *  8 +
    df["junction_known"]            *  6 +
    norm(df["Spillover_Factor"])    *  5
)
df["zone_base"] = df["police_station"].map(ZONE_BASE).fillna(60)
df["CIS"]       = (df["zone_base"] + row_adjustment).clip(0, 100).round(1)
df["CIS_Tier"]  = np.select(
    [df["CIS"] >= 85, df["CIS"] >= 70, df["CIS"] >= 55],
    ["CRITICAL", "HIGH", "MEDIUM"], default="LOW"
)

df.to_csv("data/violations_features.csv", index=False)
print(f"✅  Features: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"    CIS tiers: {df['CIS_Tier'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════
# SECTION 4 — DBSCAN Clustering
# ════════════════════════════════════════════════════════════════
print("\n⚙   DBSCAN clustering …")
df_geo = df.dropna(subset=["latitude", "longitude"]).copy()
coords = np.radians(df_geo[["latitude", "longitude"]].values)

db = DBSCAN(eps=0.10/6371.0, min_samples=5,
            algorithm="ball_tree", metric="haversine")
df_geo["cluster_id"] = db.fit_predict(coords)
df_geo = df_geo[df_geo["cluster_id"] != -1].copy()
print(f"    {df_geo['cluster_id'].nunique()} micro-hotspot clusters discovered")

cluster_summary = (
    df_geo.groupby("cluster_id").agg(
        center_lat        = ("latitude",         "mean"),
        center_lon        = ("longitude",        "mean"),
        total_violations  = ("id",               "count"),
        primary_violation = ("violation_type",   lambda x: x.mode()[0] if len(x) else "UNKNOWN"),
        mean_cis          = ("CIS",              "mean"),
        mean_oi           = ("Obstruction_Index","mean"),
        police_station    = ("police_station",   "first"),
        junction_name     = ("junction_name",    lambda x:
                              x[x != "No Junction"].mode()[0]
                              if (x != "No Junction").any() else "No Junction"),
    ).reset_index()
)
cluster_summary["CIS_Tier"] = np.select(
    [cluster_summary["mean_cis"] >= 85,
     cluster_summary["mean_cis"] >= 70,
     cluster_summary["mean_cis"] >= 55],
    ["CRITICAL", "HIGH", "MEDIUM"], default="LOW"
)
cluster_summary["fine_recovery"] = (
    cluster_summary["total_violations"] *
    cluster_summary["mean_cis"].apply(lambda c: 1000 if c>=85 else 500 if c>=70 else 200)
).astype(int)

cluster_summary.to_csv("data/cluster_map_data.csv", index=False)
print(f"✅  Cluster map → data/cluster_map_data.csv")

# ════════════════════════════════════════════════════════════════
# SECTION 5 — XGBoost 4-hour Shift Forecast
# ════════════════════════════════════════════════════════════════
print("\n⚙   Building shift grid + training XGBoost …")

df_geo["time_floor_4h"] = df_geo["created_datetime"].dt.floor("4h")

shift_grid_model = (
    df_geo.groupby(["cluster_id", "time_floor_4h"]).agg(
        Total_Footprint = ("Physical_Footprint", "sum"),
        Violation_Count = ("id",                 "count"),
    ).reset_index()
)

all_clusters = shift_grid_model["cluster_id"].unique()
all_times    = pd.date_range(
    start=shift_grid_model["time_floor_4h"].min(),
    end  =shift_grid_model["time_floor_4h"].max(),
    freq ="4h",
)
full_index = pd.MultiIndex.from_product(
    [all_clusters, all_times], names=["cluster_id", "time_floor_4h"]
)
shift_grid_model = (
    shift_grid_model
    .set_index(["cluster_id", "time_floor_4h"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

shift_grid_model["hour"]        = shift_grid_model["time_floor_4h"].dt.hour
shift_grid_model["day_of_week"] = shift_grid_model["time_floor_4h"].dt.dayofweek
shift_grid_model["Is_Weekend"]  = (shift_grid_model["day_of_week"] >= 5).astype(int)
shift_grid_model["month"]       = shift_grid_model["time_floor_4h"].dt.month

shift_grid_model = shift_grid_model.sort_values(["cluster_id", "time_floor_4h"])
shift_grid_model["Lag_1_Shift"] = shift_grid_model.groupby("cluster_id")["Total_Footprint"].shift(1)
shift_grid_model["Lag_1_Day"]   = shift_grid_model.groupby("cluster_id")["Total_Footprint"].shift(6)
shift_grid_model["Lag_1_Week"]  = shift_grid_model.groupby("cluster_id")["Total_Footprint"].shift(42)
shift_grid_model["Target"]      = shift_grid_model.groupby("cluster_id")["Total_Footprint"].shift(-1)

model_data = shift_grid_model.dropna().copy()

X = model_data[FEATURES]
y = np.log1p(model_data["Target"])

split   = int(len(X) * 0.8)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]

xgb_model = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.80, reg_alpha=0.10,
    random_state=42, n_jobs=-1,
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

y_pred_log  = xgb_model.predict(X_te)
y_pred_real = np.expm1(y_pred_log)
y_test_real = np.expm1(y_te)

mae  = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
print(f"\n📊  MODEL EVALUATION")
print(f"    MAE  : {mae:.4f}")
print(f"    RMSE : {rmse:.4f}")

# Dispatch queue
model_data = model_data.copy()
model_data["Predicted_Next"] = np.expm1(xgb_model.predict(X))

current_state = model_data.groupby("cluster_id").last().reset_index()
current_state["Predicted_Next"] = np.expm1(xgb_model.predict(current_state[FEATURES]))

dispatch_queue = (
    current_state.sort_values("Predicted_Next", ascending=False)
    .merge(cluster_summary[[
        "cluster_id","center_lat","center_lon",
        "primary_violation","police_station",
        "total_violations","mean_cis","CIS_Tier",
        "fine_recovery","junction_name",
    ]], on="cluster_id", how="left")
)

# Save all outputs
fi_df = pd.DataFrame({"feature": FEATURES, "importance": xgb_model.feature_importances_})
fi_df.to_csv("data/feature_importance.csv", index=False)
dispatch_queue.to_csv("data/dispatch_queue.csv",     index=False)
model_data.to_csv("data/model_grid.csv",             index=False)
shift_grid_model.to_csv("data/micro_shift_grid.csv", index=False)

# Save eval data
# for visualisation notebook
eval_df = pd.DataFrame({
    "Real_Footprint":      y_test_real.values,
    "Predicted_Footprint": y_pred_real,
})
eval_df.to_csv("data/eval_results.csv", index=False)

# Station → cluster map (used by hotspot matrix)
cluster_station_map = df_geo.groupby("cluster_id")["police_station"].first().reset_index()
cluster_station_map.to_csv("data/cluster_station_map.csv", index=False)

joblib.dump(xgb_model, "models/dispatch_model.pkl")
joblib.dump({
    "feature_names":    FEATURES,
    "mae":              mae,
    "rmse":             rmse,
    "vehicle_mass_map": VEHICLE_MASS_MAP,
    "zone_base":        ZONE_BASE,
    "adjacency":        ADJACENCY,
}, "models/metadata.pkl")

print(f"\n✅  All outputs saved.")
print("\n" + "=" * 60)
print("Pipeline complete. Next steps:")
print("  streamlit run app_streamlit.py")
print("  streamlit run visualizations.py")
print("=" * 60)