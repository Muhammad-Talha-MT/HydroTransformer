# train_simple.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import argparse
import random
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.hydro_transformer import CNNTransformer
from models.cnn_lstm import CNNLSTM

# ---- your optimized dataset / sampler / collate ----
from data_loader_temporal import (
    WatershedFlowDataset,
    GroupedBatchSampler,
    per_ws_collate_optimized,
)

# =========================
# Minimal config (edit here)
# =========================
H5_PATH   = "/data/HydroTransformer/daymet/processed_daymet_watersheds_clipped.h5"
CSV_PATH  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/discharge.csv"
STATIC_H5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

TRAIN_START_YEAR = 2000
TRAIN_END_YEAR   = 2013
VAL_START_YEAR   = 2000
VAL_END_YEAR     = 2013

SEQ_LEN    = 365
LEAD_DAYS  = 1
STRIDE     = 30

BATCH_SIZE  = 8
NUM_WORKERS = 4
PIN_MEMORY  = True
EPOCHS      = 1500
LR          = 1e-5
WEIGHT_DECAY = 0.0
SEED = 123

VARIABLES = ["prcp","tmin","tmax","srad","vp","dayl","swe"]

TRAIN_WATERSHEDS = [
    4096405, 4101500, 4124500, 4106000, 4152238, 4164300, 4176500, 4118500,
    4157005, 4105700, 4159900, 4097540, 4126970, 4122200, 4102700, 4112000,
    4101800, 4099000, 4166500, 4113000, 4122100, 4160600, 4151500, 4167000,
    4108800, 4104945, 4121500, 4121300, 4109000, 4117500, 4124200, 4175600,
    4147500, 4105500, 4122500, 4164100, 4146063, 4142000, 4121970, 4148500,
    4096515, 4097500, 4154000, 4146000, 4125550, 4116000, 4121944
]

VAL_WATERSHEDS = [
    4159492, 4115000, 4115265, 4112500, 4176000, 4114000, 4102500, 4148140, 4108600, 4105000, 4163400, 4144500
]
# ALL_WATERSHEDS = [4157005]
RUN_TAG        = "cnnlstm_spatial_t2013_59ws_365seq"
OUT_MODELS_DIR = f"../models/{RUN_TAG}/"
OUT_PLOTS_DIR  = f"../plots/{RUN_TAG}/"
TB_DIR         = f"../runs/{RUN_TAG}/"
LOG_FILE       = f"../logs/{RUN_TAG}.log"
SCALER_DIR     = f"../scalers/{RUN_TAG}/"

os.makedirs(OUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# =========================
# Logger (file only)
# =========================
def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("TrainRun")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = RotatingFileHandler(log_file, mode="a", maxBytes=10_000_000, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    logger.propagate = False
    return logger

logger = setup_logger(LOG_FILE)
logger.info("Logger initialized.")

# =========================
# Repro & device
# =========================
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED)

# =========================
# Datasets / Loaders
# =========================
def _make_dataset(ws_list, start_year, end_year):
    return WatershedFlowDataset(
        h5_path=H5_PATH,
        csv_path=CSV_PATH,
        static_h5=STATIC_H5,
        variables=VARIABLES,
        seq_len=SEQ_LEN, stride=STRIDE, lead_days=LEAD_DAYS,
        start_year=start_year, end_year=end_year, watersheds=ws_list,
    )

def make_loaders():
    train_ds = _make_dataset(TRAIN_WATERSHEDS, TRAIN_START_YEAR, TRAIN_END_YEAR)
    val_ds   = _make_dataset(VAL_WATERSHEDS, VAL_START_YEAR,   VAL_END_YEAR)

    print(f"Dataset sizes: train={len(train_ds)} | val={len(val_ds)}")
    logger.info(f"Dataset sizes: train={len(train_ds)} | val={len(val_ds)}")

    train_sampler = GroupedBatchSampler(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_sampler   = GroupedBatchSampler(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    common = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=2,
        persistent_workers=(NUM_WORKERS > 0),
        collate_fn=per_ws_collate_optimized,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **common)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,   **common)
    C = train_ds[0]["X"].shape[1]
    return train_loader, val_loader, C

# =========================
# NSE utilities
# =========================
def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs.astype(np.float64); pred = pred.astype(np.float64)
    denom = np.sum((obs - obs.mean())**2)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    num = np.sum((obs - pred)**2)
    return 1.0 - (num / denom)

@torch.no_grad()
def compute_watershed_nse(model, loader) -> dict:
    model.eval()
    per_ws = {}
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True).float()
        torch.nan_to_num_(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_true = batch["y"][:, 0].detach().cpu().numpy()
        y_pred = model(X).detach().cpu().numpy().reshape(-1)
        for i, m in enumerate(batch["meta"]):
            ws = m["watershed"]
            d = per_ws.setdefault(ws, {"obs": [], "pred": []})
            d["obs"].append(float(y_true[i]))
            d["pred"].append(float(y_pred[i]))
    nse_by_ws = {}
    for ws, vals in per_ws.items():
        obs = np.asarray(vals["obs"], dtype=np.float64)
        pred = np.asarray(vals["pred"], dtype=np.float64)
        nse_ws = nse(obs, pred)
        if np.isfinite(nse_ws):
            nse_by_ws[ws] = float(nse_ws)
    return nse_by_ws

def summarize_and_log_nse(nse_by_ws: dict, split: str, epoch: int, writer: SummaryWriter | None):
    if not nse_by_ws:
        msg = f"[Epoch {epoch}] ({split}) NSE: mean=nan | median=nan | n_ws=0"
        print(msg); logger.info(msg)
        if writer:
            writer.add_scalar(f"{split}/NSE_mean", float('nan'), epoch)
            writer.add_scalar(f"{split}/NSE_median", float('nan'), epoch)
        return float('nan'), float('nan')
    vals = np.array(list(nse_by_ws.values()), dtype=np.float64)
    mean_nse = float(np.nanmean(vals))
    median_nse = float(np.nanmedian(vals))
    msg = f"[Epoch {epoch}] ({split}) NSE: mean={mean_nse:.4f} | median={median_nse:.4f} | n_ws={len(nse_by_ws)}"
    print(msg); logger.info(msg)
    if writer:
        writer.add_scalar(f"{split}/NSE_mean", mean_nse, epoch)
        writer.add_scalar(f"{split}/NSE_median", median_nse, epoch)
    return mean_nse, median_nse

def log_val_per_watershed_nse_to_file(nse_by_ws: dict, epoch: int):
    if (epoch % 5) != 0 or not nse_by_ws:
        return
    items = sorted(nse_by_ws.items(), key=lambda kv: kv[0])
    logger.info(f"[Epoch {epoch}] (val) Per-watershed NSE (count={len(items)}):")
    for ws, v in items:
        logger.info(f"  WS {ws}: NSE={v:.6f}")

# =========================
# Checkpoint helpers
# =========================
def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

def _maybe_fix_dataparallel_keys(state_dict: dict, target_is_dp: bool) -> dict:
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if target_is_dp and not has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    if not target_is_dp and has_module:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

def save_checkpoint(path: str, epoch: int, model: nn.Module, opt, best_val: float,
                    global_step_train: int, global_step_val: int):
    state = {
        "epoch": epoch,
        "model_state": _unwrap(model).state_dict(),
        "optimizer_state": opt.state_dict(),
        "best_val": best_val,
        "global_step_train": global_step_train,
        "global_step_val": global_step_val,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(state, path)
    logger.info(f"Checkpoint saved -> {path}")

def get_base_model(model):
    """Return the underlying model if wrapped in DataParallel, else the model itself."""
    return model.module if isinstance(model, nn.DataParallel) else model


def load_checkpoint(path, model, optimizer=None, scaler=None,
                    map_location=None, strict=True, load_optim=True):
    """
    Simple, robust checkpoint loader that handles 'module.' prefixes and
    different key names in the checkpoint dict.
    """
    map_location = map_location or 'cpu'

    # PyTorch 2.6+: explicitly disable weights_only safety mode
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch that doesn't know weights_only
        ckpt = torch.load(path, map_location=map_location)

    # ---- 1) find the model state dict key ----
    state_key = None
    for k in ("model_state", "model_state_dict", "state_dict"):
        if k in ckpt:
            state_key = k
            break
    if state_key is None:
        raise KeyError(
            f"No model_state / model_state_dict / state_dict found in checkpoint {path}"
        )

    state_dict = ckpt[state_key]

    # ---- 2) fix 'module.' prefix mismatch between ckpt and base model ----
    base_model = get_base_model(model)
    base_state = base_model.state_dict()

    ckpt_has_module = any(k.startswith("module.") for k in state_dict.keys())
    base_has_module = any(k.startswith("module.") for k in base_state.keys())

    if ckpt_has_module and not base_has_module:
        # strip "module." from keys
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not ckpt_has_module and base_has_module:
        # add "module." to keys (rare case)
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    # ---- 3) load into the (unwrapped) model ----
    base_model.load_state_dict(state_dict, strict=strict)

    # ---- 4) load optimizer state if requested ----
    start_epoch = int(ckpt.get('epoch', 0))

    if optimizer is not None and load_optim:
        opt_key = None
        for k in ("optimizer_state", "optimizer_state_dict"):
            if k in ckpt:
                opt_key = k
                break
        if opt_key is not None:
            try:
                optimizer.load_state_dict(ckpt[opt_key])
            except ValueError as e:
                print(f"[warn] Optimizer state incompatible ({e}). Continuing with a fresh optimizer.")

    # ---- 5) load scaler (optional, if you ever add AMP) ----
    if scaler is not None and 'scaler_state' in ckpt:
        try:
            scaler.load_state_dict(ckpt['scaler_state'])
        except Exception:
            print("[warn] AMP scaler load failed; continuing fresh scaler.")

    print(f"Loaded checkpoint '{os.path.basename(path)}' at epoch {start_epoch}.")
    return start_epoch
# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, loss_fn, opt, epoch, writer: SummaryWriter | None, global_step: int):
    model.train()
    total, steps = 0.0, 0
    pbar = tqdm(loader, desc=f"Train {epoch}", ncols=90, leave=False)
    for batch in pbar:
        X = batch["X"].to(device, non_blocking=True).float()
        torch.nan_to_num_(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = batch["y"][:, 0].to(device, non_blocking=True).unsqueeze(1).float()
  
        opt.zero_grad(set_to_none=True)
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()

        total += float(loss.detach().item()); steps += 1
        avg = total / steps
        pbar.set_postfix(loss=avg)

        if writer is not None:
            writer.add_scalar("train/step_loss", float(loss.detach().item()), global_step)
            global_step += 1

    epoch_loss = total / max(1, steps)
    if writer is not None:
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
    logger.info(f"[Epoch {epoch}] train_epoch_loss={epoch_loss:.6f}")
    return epoch_loss, global_step

@torch.no_grad()
def eval_loss(model, loader, loss_fn, epoch, writer: SummaryWriter | None, global_step_val: int):
    model.eval()
    total, steps = 0.0, 0
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True).float()
        torch.nan_to_num_(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = batch["y"][:, 0].to(device, non_blocking=True).unsqueeze(1).float()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        total += float(loss.detach().item()); steps += 1
        if writer is not None:
            writer.add_scalar("val/step_loss", float(loss.detach().item()), global_step_val)
            global_step_val += 1

    val_loss = total / max(1, steps)
    if writer is not None:
        writer.add_scalar("val/epoch_loss", val_loss, epoch)
    logger.info(f"[Epoch {epoch}] val_epoch_loss={val_loss:.6f}")
    return val_loss, global_step_val

# =========================
# Main
# =========================
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint (.pth) to resume from")
    return p

def main():
    args = build_argparser().parse_args()

    os.makedirs(TB_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=TB_DIR)

    train_loader, val_loader, C = make_loaders()
    print(f"Input channels: {C}")
    logger.info(f"Input channels: {C}")

    # --- Choose model ---
    model = CNNLSTM(C, cnn_width=32, lstm_hidden=128, lstm_layers=2, out_horizons=1).to(device)
    # model = CNNTransformer(...).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    # ---- Resume (only flag) ----
    start_epoch = 0
    best_val = float('inf')
    global_step_train = 0
    global_step_val = 0

    if args.resume is not None and os.path.isfile(args.resume):
        try:
            start_epoch = load_checkpoint(
                args.resume,
                model,
                optimizer=opt,
                map_location=device,
                strict=True,
                load_optim=True,
            )
            start_epoch = max(0, start_epoch)
            logger.info(f"Resumed from {args.resume}. Starting at epoch {start_epoch + 1}.")
        except Exception as e:
            logger.info(f"Failed to resume from {args.resume}: {e}")

    # ---- Train loop ----
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        tr_loss, global_step_train = train_one_epoch(model, train_loader, loss_fn, opt, epoch, writer, global_step_train)
        va_loss, global_step_val   = eval_loss(model, val_loader, loss_fn, epoch, writer, global_step_val)

        nse_val = compute_watershed_nse(model, val_loader)
        summarize_and_log_nse(nse_val, split="val", epoch=epoch, writer=writer)
        log_val_per_watershed_nse_to_file(nse_val, epoch)

        print(f"[Epoch {epoch}] train={tr_loss:.4f} | val={va_loss:.4f}")
        logger.info(f"[Epoch {epoch}] train={tr_loss:.6f} | val={va_loss:.6f}")

        if va_loss < best_val:
            best_val = va_loss
            save_checkpoint(
                os.path.join(OUT_MODELS_DIR, "best_hydrotransformer.pth"),
                epoch, model, opt, best_val, global_step_train, global_step_val
            )
            logger.info(f"Updated best model at epoch {epoch} (val={va_loss:.6f}).")

    # Final save
    save_checkpoint(
        os.path.join(OUT_MODELS_DIR, "last.pth"),
        EPOCHS, model, opt, best_val, global_step_train, global_step_val
    )
    writer.close()

if __name__ == "__main__":
    main()
