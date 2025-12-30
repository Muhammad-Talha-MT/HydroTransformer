# evaluation_spatial.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import math
import random
import json
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

# ---- models & data loader (same as training) ----
from models.hydro_transformer_film import ViTTemporalFusion
from models.hydro_transformer import ViTTemporalNoStatic
from data_loader_temporal import (
    WatershedFlowDataset,
    GroupedBatchSampler,
    per_ws_collate_optimized,
)

# =======================================================
# Config (edit these to match training & test split)
# =======================================================
H5_PATH   = "/data/HydroTransformer/daymet/processed_daymet_watersheds_clipped.h5"
CSV_PATH  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/discharge.csv"
STATIC_H5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

# ---- per-watershed flow normalization stats ----
FLOW_SCALE_JSON = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/flow_scalers.json"

VARIABLES = ["prcp","tmin","tmax","srad","vp","dayl","swe"]

SEQ_LEN    = 365
LEAD_DAYS  = 1
STRIDE     = 1

BATCH_SIZE   = 4
NUM_WORKERS  = 4
PIN_MEMORY   = True

# ----- TEST / VAL split (edit as needed) -----
TEST_START_YEAR = 2014
TEST_END_YEAR   = 2016 # inclusive

TEST_WATERSHEDS = [
    4096405, 4096515, 4097500, 4097540, 4099000, 4101500, 4101800, 4102500,
    4102700, 4104945, 4105000, 4105500, 4105700, 4106000, 4108600, 4108800,
    4109000, 4112000, 4112500, 4113000, 4114000, 4115000, 4115265, 4116000,
    4117500, 4118500, 4121300, 4121500, 4121944, 4121970, 4122100, 4122200,
    4122500, 4124200, 4124500, 4125550, 4126970, 4142000, 4176500, 4176000,
    4144500, 4146000, 4146063, 4147500, 4148140, 4148500, 4151500, 4152238,
    4154000, 4157005, 4159492, 4159900, 4160600, 4163400, 4164100, 4164300,
    4166500, 4167000, 4175600
]

# TEST_WATERSHEDS = [
#     4159492, 4115000, 4115265, 4112500, 4176000, 4114000, 4102500, 4148140,
#     4108600, 4105000, 4163400, 4144500
# ]

# Year for single-year hydrograph
YEAR_FOR_DETAILED_HYDRO = 2014

# Base offset (days) from dataset start to first target day
# For sliding window: first target at index (SEQ_LEN - 1 + LEAD_DAYS - 1)
BASE_OFFSET_DAYS = (SEQ_LEN - 1) + (LEAD_DAYS - 1)

# ----- Model hyperparams (must match training) -----
FUSION          = "film"  # "film", "concat", "prefix", "late", "dual" etc.
PATCH_SIZE      = 8
D_MODEL         = 192
SPATIAL_LAYERS  = 4
SPATIAL_HEADS   = 6
TEMPORAL_LAYERS = 2
TEMPORAL_HEADS  = 6
PREFIX_TOKENS   = 2

# ----- Checkpoint & output dirs -----
RUN_TAG        = "vitfusion_temporal_t2013_59ws_365seq_nostatic"
CKPT_PATH      = f"../models/{RUN_TAG}/best_hydrotransformer.pth"

EVAL_OUT_DIR        = f"../eval/{RUN_TAG}/"
HYDRO_OUT_DIR       = os.path.join(EVAL_OUT_DIR, "hydrographs")                # full-series (index-based)
SCATTER_OUT_DIR     = os.path.join(EVAL_OUT_DIR, "scatter_plots")
GLOBAL_FIGS_OUT_DIR = os.path.join(EVAL_OUT_DIR, "global")

# New: extra outputs
YEARLY_HYDRO_OUT_DIR = os.path.join(EVAL_OUT_DIR, f"hydrographs_year{YEAR_FOR_DETAILED_HYDRO}")
CLIM_HYDRO_OUT_DIR   = os.path.join(EVAL_OUT_DIR, "hydrographs_climatology")

os.makedirs(HYDRO_OUT_DIR, exist_ok=True)
os.makedirs(SCATTER_OUT_DIR, exist_ok=True)
os.makedirs(GLOBAL_FIGS_OUT_DIR, exist_ok=True)
os.makedirs(YEARLY_HYDRO_OUT_DIR, exist_ok=True)
os.makedirs(CLIM_HYDRO_OUT_DIR, exist_ok=True)

SEED = 123

# =======================================================
# Utils
# =======================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# ---- load FLOW_SCALE from JSON (global) ----
if not os.path.isfile(FLOW_SCALE_JSON):
    raise FileNotFoundError(f"FLOW_SCALE_JSON not found at: {FLOW_SCALE_JSON}")
with open(FLOW_SCALE_JSON, "r") as f:
    FLOW_SCALE = json.load(f)


def normalize_ws_id(ws_id):
    """
    Convert things like:
      4102500, "4102500", "4102500_watershed"
    into the plain numeric string "4102500".
    """
    s = str(ws_id)
    m = re.search(r"\d+", s)
    if not m:
        raise KeyError(f"Could not extract numeric watershed id from: {ws_id}")
    return m.group(0)


def get_ws_mean_std(ws_id):
    """
    Given a watershed ID (int or str), return (mean, std)
    using FLOW_SCALE JSON (supports either 'std' or 'var').
    Handles ids like '4102500_watershed' by stripping suffix.
    """
    key = normalize_ws_id(ws_id)

    if key not in FLOW_SCALE:
        raise KeyError(f"Watershed {ws_id} (normalized to {key}) not found in FLOW_SCALE stats JSON.")

    stats = FLOW_SCALE[key]
    if "std" in stats:
        std = float(stats["std"])
    elif "var" in stats:
        std = float(math.sqrt(stats["var"]))
    else:
        raise KeyError(
            f"Stats for watershed {ws_id} must contain 'std' or 'var'; got keys {list(stats.keys())}"
        )
    mean = float(stats["mean"])
    return mean, std

# -------------------------------------------------------
# Static fusion helper (copied from training)
# -------------------------------------------------------
def build_static_tensor_from_batch(batch, device, dtype=torch.float32):
    """
    Build S=(B, C_static, H, W) from batch statics.
    Standard layout: [DEM, awc, fc, soil[0], soil[1], soil[2]]
    """
    X = batch["X"]
    B, L, C, H, W = X.shape

    def _prep_one(t, expect_c=1):
        if t is None:
            return torch.zeros(expect_c, H, W, dtype=dtype, device=device)
        t = t.to(device=device, dtype=dtype)
        if t.ndim == 2:
            t = t.unsqueeze(0)  # (1,Hs,Ws)
        elif t.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected static tensor ndim={t.ndim}")
        # pad/truncate channels
        if t.shape[0] < expect_c:
            pad = torch.zeros(expect_c - t.shape[0], *t.shape[1:], dtype=dtype, device=device)
            t = torch.cat([t, pad], dim=0)
        elif t.shape[0] > expect_c:
            t = t[:expect_c]
        # resize
        t = F.interpolate(t.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
        return t

    DEM  = _prep_one(batch.get("DEM"),  expect_c=1)
    awc  = _prep_one(batch.get("awc"),  expect_c=1)
    fc   = _prep_one(batch.get("fc"),   expect_c=1)
    soil = _prep_one(batch.get("soil"), expect_c=3)

    S_chw = torch.cat([DEM, awc, fc, soil], dim=0)  # (6, H, W)
    S = S_chw.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # (B, 6, H, W)
    torch.nan_to_num_(S, nan=0.0, posinf=0.0, neginf=0.0)
    return S

# -------------------------------------------------------
# Dataloader for TEST set
# -------------------------------------------------------
def make_test_loader():
    test_ds = WatershedFlowDataset(
        h5_path=H5_PATH,
        csv_path=CSV_PATH,
        static_h5=STATIC_H5,
        variables=VARIABLES,
        seq_len=SEQ_LEN,
        stride=STRIDE,
        lead_days=LEAD_DAYS,
        start_year=TEST_START_YEAR,
        end_year=TEST_END_YEAR,
        watersheds=TEST_WATERSHEDS,
    )
    print(f"Test dataset size: {len(test_ds)}")

    test_sampler = GroupedBatchSampler(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    common = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=2,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=per_ws_collate_optimized,
    )
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler, **common)

    # Channel counts
    sample = test_ds[0]
    C_dyn = sample["X"].shape[1]
    C_static = 6  # DEM + awc + fc + 3 soil
    return test_loader, C_dyn, C_static

# -------------------------------------------------------
# Metrics: NSE, KGE, RMSE
# -------------------------------------------------------
def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs.astype(np.float64)
    pred = pred.astype(np.float64)
    denom = np.sum((obs - obs.mean())**2)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    num = np.sum((obs - pred)**2)
    return 1.0 - (num / denom)

def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs.astype(np.float64)
    pred = pred.astype(np.float64)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    if obs.size < 2:
        return np.nan
    mu_o = obs.mean()
    mu_p = pred.mean()
    std_o = obs.std(ddof=1)
    std_p = pred.std(ddof=1)

    if std_o == 0:
        return np.nan

    r = np.corrcoef(obs, pred)[0, 1]
    alpha = std_p / std_o if std_o != 0 else np.nan
    beta = mu_p / mu_o if mu_o != 0 else np.nan

    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan

    return 1.0 - math.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)

def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs.astype(np.float64)
    pred = pred.astype(np.float64)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    if obs.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((obs - pred) ** 2)))

def compute_all_metrics(obs: np.ndarray, pred: np.ndarray):
    return {
        "nse": nse(obs, pred),
        "kge": kge(obs, pred),
        "rmse": rmse(obs, pred),
    }

# -------------------------------------------------------
# Load model weights (fixed for PyTorch 2.6 weights_only issue)
# -------------------------------------------------------
def load_model(model: nn.Module, ckpt_path: str, device):
    # Explicitly set weights_only=False for PyTorch 2.6+
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # For older PyTorch versions without weights_only arg
        ckpt = torch.load(ckpt_path, map_location=device)

    state_key = None
    for k in ("model_state", "model_state_dict", "state_dict"):
        if k in ckpt:
            state_key = k
            break
    if state_key is None:
        raise KeyError(f"No model_state / model_state_dict / state_dict in {ckpt_path}")

    state_dict = ckpt[state_key]

    # Handle DataParallel vs non-DataParallel mismatch
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    base_state = base_model.state_dict()
    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())
    base_has_module = any(k.startswith("module.") for k in base_state.keys())

    if ckpt_has_module and not base_has_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not ckpt_has_module and base_has_module:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    base_model.load_state_dict(state_dict, strict=True)
    print(f"Loaded weights from {os.path.basename(ckpt_path)}")

# -------------------------------------------------------
# Evaluation loop (denormalizes per-watershed)
# -------------------------------------------------------
@torch.no_grad()
def run_evaluation(model, loader):
    model.eval()
    # ws keys are normalized numeric strings like "4102500"
    per_ws_series = {}   # ws -> {"obs": [], "pred": []} (DENORMALIZED)
    all_obs = []         # global denorm obs
    all_pred = []        # global denorm pred

    pbar = tqdm(loader, desc="Evaluating", ncols=90)
    for batch in pbar:
        X = batch["X"].to(device, non_blocking=True).float()
        torch.nan_to_num_(X, nan=0.0, posinf=0.0, neginf=0.0)
        S = build_static_tensor_from_batch(batch, device=device, dtype=torch.float32)

        # Normalized values
        y_true = batch["y"][:, 0].to(device, non_blocking=True).float()
        # y_hat  = model(X, S).view(-1)
        y_hat  = model(X).view(-1)

        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_hat.detach().cpu().numpy()

        # Loop over samples, denormalize per-watershed, and store
        for i, meta in enumerate(batch["meta"]):
            raw_ws = meta["watershed"]      # e.g. "4102500_watershed"
            ws = normalize_ws_id(raw_ws)    # e.g. "4102500"

            mean_ws, std_ws = get_ws_mean_std(ws)

            y_true_denorm = y_true_np[i] * std_ws + mean_ws
            y_pred_denorm = y_pred_np[i] * std_ws + mean_ws

            d = per_ws_series.setdefault(ws, {"obs": [], "pred": []})
            d["obs"].append(float(y_true_denorm))
            d["pred"].append(float(y_pred_denorm))

            all_obs.append(float(y_true_denorm))
            all_pred.append(float(y_pred_denorm))

    all_obs = np.asarray(all_obs, dtype=np.float64)
    all_pred = np.asarray(all_pred, dtype=np.float64)

    # Per-watershed metrics (denormalized)
    per_ws_metrics = {}
    for ws, series in per_ws_series.items():
        obs = np.asarray(series["obs"], dtype=np.float64)
        pred = np.asarray(series["pred"], dtype=np.float64)
        per_ws_metrics[ws] = compute_all_metrics(obs, pred)

    # Global metrics over all samples (denormalized)
    global_metrics = compute_all_metrics(all_obs, all_pred)

    return per_ws_series, per_ws_metrics, all_obs, all_pred, global_metrics

# -------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------
def plot_hydrograph(ws, obs, pred, metrics, out_dir):
    t = np.arange(len(obs))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, obs, label="Observed", linewidth=1.5)
    ax.plot(t, pred, label="Simulated", linewidth=1.2)
    ax.set_title(f"Hydrograph - Watershed {ws}")
    ax.set_xlabel("Sample index (test set order)")
    ax.set_ylabel("Streamflow")
    ax.legend()

    txt = (
        f"NSE = {metrics['nse']:.3f}\n"
        f"KGE = {metrics['kge']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3f}"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", alpha=0.5)
    )

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"hydrograph_ws{ws}.png"), dpi=200)
    plt.close(fig)


def plot_scatter(ws_label, obs, pred, metrics, out_dir, fname_prefix):
    obs = np.asarray(obs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]

    if obs.size == 0:
        return

    # 1:1 line
    vmin = min(obs.min(), pred.min())
    vmax = max(obs.max(), pred.max())

    # Regression line
    if obs.size >= 2:
        slope, intercept = np.polyfit(obs, pred, 1)
    else:
        slope, intercept = 1.0, 0.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs, pred, s=10, alpha=0.6, label="Samples")

    x_line = np.linspace(vmin, vmax, 100)
    ax.plot(x_line, x_line, linestyle="--", linewidth=1.0, label="1:1 line")
    ax.plot(x_line, slope * x_line + intercept, linewidth=1.0, label="Regression")

    ax.set_title(f"Obs vs Pred - {ws_label}")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.legend()

    txt = (
        f"NSE = {metrics['nse']:.3f}\n"
        f"KGE = {metrics['kge']:.3f}\n"
        f"RMSE = {metrics['rmse']:.3f}"
    )
    ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", alpha=0.5)
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{fname_prefix}_{ws_label}.png")
    out_path = out_path.replace(" ", "_")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# -------------------------------------------------------
# New: Single-year hydrograph (e.g., 2013)
# -------------------------------------------------------
def plot_single_year_hydrograph(ws, df, year, out_dir):
    """
    df columns: ['year','doy','obs','pred']
    Plots obs/pred for a specific year vs day-of-year.
    """
    df_year = df[df["year"] == year].copy()
    if df_year.empty:
        print(f"[WARN] No data for watershed {ws} in year {year}, skipping single-year hydrograph.")
        return

    df_year = df_year.sort_values("doy")
    t = df_year["doy"].values
    obs = df_year["obs"].values
    pred = df_year["pred"].values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, obs, label="Observed", linewidth=1.5)
    ax.plot(t, pred, label="Simulated", linewidth=1.2)

    ax.set_title(f"Hydrograph - Watershed {ws} - Year {year}")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Streamflow")
    ax.legend()

    # Put month labels on x-axis (365-day synthetic year)
    month_starts = pd.date_range("2001-01-01", "2001-12-31", freq="MS")
    xticks = [d.timetuple().tm_yday for d in month_starts]
    xlabels = [d.strftime("%b") for d in month_starts]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"hydrograph_ws{ws}_year{year}.png"), dpi=200)
    plt.close(fig)

# -------------------------------------------------------
# New: Climatological (multi-year mean) hydrograph
# -------------------------------------------------------
def plot_climatology_hydrograph(ws, df, out_dir, start_year=None, end_year=None):
    """
    df columns: ['year','doy','obs','pred']
    Builds a Jan–Dec hydrograph:
      - obs_mean(doy), obs_mean ± obs_std
      - pred_mean(doy), pred_mean ± pred_std
    """
    if df.empty:
        print(f"[WARN] Empty DF for watershed {ws}, skipping climatology.")
        return

    if (start_year is not None) and (end_year is not None):
        df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    if df.empty:
        print(f"[WARN] No data for watershed {ws} in {start_year}-{end_year}, skipping climatology.")
        return

    grouped = df.groupby("doy").agg(
        obs_mean=("obs", "mean"),
        obs_std=("obs", "std"),
        pred_mean=("pred", "mean"),
        pred_std=("pred", "std"),
        n=("obs", "count"),
    ).reset_index()

    t = grouped["doy"].values
    obs_mean = grouped["obs_mean"].values
    obs_std  = grouped["obs_std"].values
    pred_mean = grouped["pred_mean"].values
    pred_std  = grouped["pred_std"].values

    fig, ax = plt.subplots(figsize=(10, 4))

    # Observed band + mean
    ax.fill_between(
        t,
        obs_mean - obs_std,
        obs_mean + obs_std,
        alpha=0.2,
        label="Obs ±1σ",
    )
    ax.plot(t, obs_mean, linewidth=1.5, label="Obs mean")

    # Predicted band + mean
    ax.fill_between(
        t,
        pred_mean - pred_std,
        pred_mean + pred_std,
        alpha=0.2,
        label="Pred ±1σ",
    )
    ax.plot(t, pred_mean, linewidth=1.5, label="Pred mean")

    title = f"Climatological Hydrograph - Watershed {ws}"
    if (start_year is not None) and (end_year is not None):
        title += f" ({start_year}-{end_year})"
    ax.set_title(title)

    ax.set_xlabel("Day of Year (Jan–Dec)")
    ax.set_ylabel("Streamflow")

    # Month ticks
    month_starts = pd.date_range("2001-01-01", "2001-12-31", freq="MS")
    xticks = [d.timetuple().tm_yday for d in month_starts]
    xlabels = [d.strftime("%b") for d in month_starts]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.legend()
    plt.tight_layout()

    fname = f"climatology_ws{ws}"
    if (start_year is not None) and (end_year is not None):
        fname += f"_{start_year}-{end_year}"
    fig.savefig(os.path.join(out_dir, f"{fname}.png"), dpi=200)
    plt.close(fig)

# -------------------------------------------------------
# Save hydrograph data to CSV
# -------------------------------------------------------
def save_hydrographs_to_csv(per_ws_series, out_dir):
    """
    For each watershed, save a CSV with columns:
    index, obs, pred

    Values are in DENORMALIZED units.
    """
    for ws, series in per_ws_series.items():
        obs = np.asarray(series["obs"], dtype=np.float64)
        pred = np.asarray(series["pred"], dtype=np.float64)
        idx = np.arange(len(obs))
        df = pd.DataFrame({"index": idx, "obs": obs, "pred": pred})
        df.to_csv(os.path.join(out_dir, f"hydrograph_ws{ws}.csv"), index=False)

# -------------------------------------------------------
# NEW: Save metrics summary (per basin + mean/median/global) to CSV
# -------------------------------------------------------
def save_metrics_summary_to_csv(per_ws_metrics,
                                global_metrics,
                                mean_nse,
                                median_nse,
                                mean_kge,
                                median_kge,
                                out_dir):
    """
    Save per-watershed NSE/KGE/RMSE, plus mean/median over watersheds
    and global metrics, into a single CSV file.
    """
    rows = []

    # per-watershed rows
    for ws, m in sorted(per_ws_metrics.items(), key=lambda kv: kv[0]):
        rows.append({
            "watershed": str(ws),
            "scope": "per_watershed",
            "nse": float(m["nse"]),
            "kge": float(m["kge"]),
            "rmse": float(m["rmse"]),
        })

    # mean over watersheds
    rows.append({
        "watershed": "ALL_WS",
        "scope": "mean_over_watersheds",
        "nse": float(mean_nse),
        "kge": float(mean_kge),
        "rmse": float("nan"),
    })

    # median over watersheds
    rows.append({
        "watershed": "ALL_WS",
        "scope": "median_over_watersheds",
        "nse": float(median_nse),
        "kge": float(median_kge),
        "rmse": float("nan"),
    })

    # global metrics over all samples
    rows.append({
        "watershed": "ALL_WS",
        "scope": "global_over_samples",
        "nse": float(global_metrics["nse"]),
        "kge": float(global_metrics["kge"]),
        "rmse": float(global_metrics["rmse"]),
    })

    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "metrics_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved metrics summary CSV to: {out_path}")

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("Building test loader...")
    test_loader, C_dyn, C_static = make_test_loader()
    print(f"Test loader ready. Dynamic channels: {C_dyn} | Static channels: {C_static}")

    print("Building model...")
    # model = ViTTemporalFusion(
    #     c_dyn=C_dyn,
    #     c_static=C_static,
    #     fusion=FUSION,
    #     patch_size=PATCH_SIZE,
    #     d_model=D_MODEL,
    #     spatial_layers=SPATIAL_LAYERS,
    #     spatial_heads=SPATIAL_HEADS,
    #     temporal_layers=TEMPORAL_LAYERS,
    #     temporal_heads=TEMPORAL_HEADS,
    #     out_horizons=1,
    #     static_dim=D_MODEL,
    #     prefix_tokens=PREFIX_TOKENS,
    # ).to(device)

    model = ViTTemporalNoStatic(
        c_dyn=C_dyn,
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        spatial_layers=SPATIAL_LAYERS,
        spatial_heads=SPATIAL_HEADS,
        temporal_layers=TEMPORAL_LAYERS,
        temporal_heads=TEMPORAL_HEADS,
        out_horizons=1,
    ).to(device)

    model = nn.DataParallel(model, device_ids=None)  # uses all visible GPUs

    print("Loading checkpoint...")
    load_model(model, CKPT_PATH, device)

    print("Running evaluation on test set (denormalized outputs)...")
    per_ws_series, per_ws_metrics, all_obs, all_pred, global_metrics = run_evaluation(model, test_loader)

    # ---- Summary stats over watersheds ----
    nse_vals = [m["nse"] for m in per_ws_metrics.values() if np.isfinite(m["nse"])]
    kge_vals = [m["kge"] for m in per_ws_metrics.values() if np.isfinite(m["kge"])]

    mean_nse = float(np.mean(nse_vals)) if nse_vals else float("nan")
    median_nse = float(np.median(nse_vals)) if nse_vals else float("nan")
    mean_kge = float(np.mean(kge_vals)) if kge_vals else float("nan")
    median_kge = float(np.median(kge_vals)) if kge_vals else float("nan")

    print("===================================================")
    print("Per-watershed metrics (DENORMALIZED):")
    for ws, m in sorted(per_ws_metrics.items(), key=lambda kv: kv[0]):
        print(
            f"WS {ws}: NSE={m['nse']:.4f} | "
            f"KGE={m['kge']:.4f} | RMSE={m['rmse']:.4f}"
        )
    print("===================================================")
    print(
        f"Mean NSE   (over watersheds): {mean_nse:.4f}\n"
        f"Median NSE (over watersheds): {median_nse:.4f}\n"
        f"Mean KGE   (over watersheds): {mean_kge:.4f}\n"
        f"Median KGE (over watersheds): {median_kge:.4f}"
    )
    print("===================================================")
    print(
        "Global metrics over all TEST samples (DENORMALIZED):\n"
        f"  NSE  = {global_metrics['nse']:.4f}\n"
        f"  KGE  = {global_metrics['kge']:.4f}\n"
        f"  RMSE = {global_metrics['rmse']:.4f}"
    )

    # ---- Save metrics summary to CSV ----
    save_metrics_summary_to_csv(
        per_ws_metrics=per_ws_metrics,
        global_metrics=global_metrics,
        mean_nse=mean_nse,
        median_nse=median_nse,
        mean_kge=mean_kge,
        median_kge=median_kge,
        out_dir=EVAL_OUT_DIR,
    )

    # ---- Save hydrograph data ----
    print("Saving hydrographs (CSV)...")
    save_hydrographs_to_csv(per_ws_series, HYDRO_OUT_DIR)

    # ---- Per-watershed plots (full series) ----
    print("Making per-watershed hydrograph & scatter plots (full series)...")
    for ws, series in per_ws_series.items():
        obs = np.asarray(series["obs"], dtype=np.float64)
        pred = np.asarray(series["pred"], dtype=np.float64)
        metrics = per_ws_metrics[ws]

        # Full hydrograph (index-based)
        plot_hydrograph(ws, obs, pred, metrics, HYDRO_OUT_DIR)
        # Scatter
        plot_scatter(f"WS_{ws}", obs, pred, metrics, SCATTER_OUT_DIR, fname_prefix="scatter")

    # ---- Global scatter over all TEST samples ----
    print("Making global scatter plot...")
    plot_scatter("ALL_WS", all_obs, all_pred, global_metrics, GLOBAL_FIGS_OUT_DIR, fname_prefix="scatter_all")

    # ---- Build DataFrames for per-watershed seasonal/yearly plots using 365-day calendar ----
    print("Building seasonal / yearly plots based on 365-day calendar...")
    for ws, series in per_ws_series.items():
        obs = np.asarray(series["obs"], dtype=float)
        pred = np.asarray(series["pred"], dtype=float)
        n = len(obs)
        if n == 0:
            continue

        idx = np.arange(n, dtype=int)

        # offset in days from TEST_START_YEAR-01-01 to target day of each sample
        offset_days = BASE_OFFSET_DAYS + idx * STRIDE

        years = TEST_START_YEAR + (offset_days // 365)
        doys  = (offset_days % 365) + 1  # 1..365

        df = pd.DataFrame({
            "year": years,
            "doy":  doys,
            "obs":  obs,
            "pred": pred,
        })

        # Single-year hydrograph (e.g., 2013)
        plot_single_year_hydrograph(ws, df, YEAR_FOR_DETAILED_HYDRO, YEARLY_HYDRO_OUT_DIR)

        # Climatological hydrograph over TEST_START_YEAR–TEST_END_YEAR
        plot_climatology_hydrograph(ws, df, CLIM_HYDRO_OUT_DIR,
                                    start_year=TEST_START_YEAR,
                                    end_year=TEST_END_YEAR)

    print(f"Done. Outputs saved to: {EVAL_OUT_DIR}")

if __name__ == "__main__":
    main()
