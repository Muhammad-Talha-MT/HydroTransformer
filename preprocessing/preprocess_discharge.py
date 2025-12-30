# compute_watershed_scalers.py

import os
import json
import numpy as np
import pandas as pd

# ====== CONFIG: edit paths/years if needed ======
CSV_PATH = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"

TRAIN_START_YEAR = 2000
TRAIN_END_YEAR   = 2013

OUT_DIR   = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data"
OUT_JSON  = os.path.join(OUT_DIR, "flow_scalers.json")
OUT_CSV   = os.path.join(OUT_DIR, "flow_scalers.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# ====== LOAD DATA ======
df = pd.read_csv(CSV_PATH)

# Parse date and restrict to train years (to match your training normalization)
df["date"] = pd.to_datetime(df["date"])
mask = (df["date"].dt.year >= TRAIN_START_YEAR) & (df["date"].dt.year <= TRAIN_END_YEAR)
df_train = df.loc[mask].copy()

# All watershed columns = everything except "date"
ws_cols = [c for c in df_train.columns if c != "date"]

print(f"Found {len(ws_cols)} watershed columns.")
print("Example watersheds:", ws_cols[:5])

# ====== COMPUTE SCALERS ======
scalers = {}  # ws_id -> {mean, std, var}

for col in ws_cols:
    series = df_train[col].to_numpy(dtype=float)

    # Drop NaNs / infs just in case
    mask = np.isfinite(series)
    series = series[mask]
    if series.size == 0:
        print(f"Warning: watershed {col} has no finite values in train period, skipping.")
        continue

    # Population stats (ddof=0). If you used sample std before, change ddof=1.
    mean = float(series.mean())
    std  = float(series.std(ddof=0))
    var  = float(series.var(ddof=0))

    # Store with int key (watershed id)
    try:
        ws_id = int(col)
    except ValueError:
        ws_id = col  # fallback if column name is not pure int

    scalers[ws_id] = {"mean": mean, "std": std, "var": var}

print(f"Computed scalers for {len(scalers)} watersheds.")

# ====== SAVE JSON (for code) ======
with open(OUT_JSON, "w") as f:
    json.dump(scalers, f, indent=2)
print(f"Saved JSON scalers to: {OUT_JSON}")

# # ====== SAVE CSV (for inspection) ======
# scalers_df = pd.DataFrame.from_dict(scalers, orient="index")
# scalers_df.index.name = "watershed"
# scalers_df.to_csv(OUT_CSV)
# print(f"Saved CSV scalers to: {OUT_CSV}")
