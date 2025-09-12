# training_pretrained.py
import os
# 👉 No spaces in the list below
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import gc
import signal
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from hydro_loss import HydroLossNorm
from resnet_transformer import ImprovedHydroTransformer  # <-- your model

from data_loader_update import (
    WatershedFlowDataset,
    GroupedBatchSampler,
)

# =========================
# Config
# =========================
H5_PATH   = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
CSV_PATH  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"
STATIC_H5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

RUN_TAG        = "pretrained_resnet18_tformer_updated_09122025"
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

# Cache files (train-fitted; reused by val/test)
MM_CLIM = os.path.join("mm__CLIMATE__TRAIN_GLOBAL.json")
MM_FLOW = os.path.join("mm__FLOW__TRAIN_GLOBAL.json")
MM_STAT = os.path.join("mm__STATIC__TRAIN_GLOBAL.json")

# Transforms (applied BEFORE scaling)
CLIMATE_TRANSFORM_MAP = {
    # adapt keys to your HDF variable names; harmless if a key is absent
    "prcp": "log1p", "ppt": "log1p", "precip": "log1p",
    # "tmin": "identity",
    # "tmax": "identity",
}
STREAMFLOW_TRANSFORM = "log1p"

EPOCHS       = 15000
BATCH_SIZE   = 32            # small; BN is frozen for stability
NUM_WORKERS  = 8
PIN_MEMORY   = True
USE_AMP_DEF  = True         # can be toggled with --no-amp
GRAD_CLIP    = 1.0

SEQ_LEN    = 5
LEAD_DAYS  = 1
START_YEAR = 2011
END_YEAR   = 2021

# =========================
# Watershed splits (HARDCODED and consistent)
# =========================
ALL_WATERSHEDS = [
    4096405, 4096515, 4097500, 4097540, 4099000, 4101500, 4101800, 4102500,
    4102700, 4104945, 4105000, 4105500, 4105700, 4106000, 4108600, 4108800,
    4109000, 4112000, 4112500, 4113000, 4114000, 4115000, 4115265, 4116000,
    4117500, 4118500, 4121300, 4121500, 4121944, 4121970, 4122100, 4122200,
    4122500, 4124200, 4124500, 4125550, 4126740, 4126970, 4127800, 4142000,
    4144500, 4146000, 4146063, 4147500, 4148140, 4148500, 4151500, 4152238,
    4154000, 4157005, 4159492, 4159900, 4160600, 4163400, 4164100, 4164300,
    4166500, 4167000, 4175600
]
VAL_WATERSHEDS = [4113000, 4144500, 4097500, 4146000, 4104945]
TEST_WATERSHEDS = [4176500, 4176000]
TRAIN_WATERSHEDS = [w for w in ALL_WATERSHEDS if w not in set(VAL_WATERSHEDS) | set(TEST_WATERSHEDS)]

# =========================
# Logger
# =========================
def setup_logger(log_file=LOG_FILE):
    logger = logging.getLogger('PretrainedRun')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    return logger

logger = setup_logger()

# =========================
# Utils
# =========================
def get_base_model(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m

def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs.astype(np.float64); pred = pred.astype(np.float64)
    denom = np.sum((obs - obs.mean())**2)
    if denom <= 0: return np.nan
    return 1.0 - np.sum((obs - pred)**2) / denom

def _denorm_streamflow(v: np.ndarray, ws: str, gmm_stream: dict) -> np.ndarray:
    """
    Denormalize using GLOBAL stats if present; otherwise fall back to per-WS.
    """
    mm = gmm_stream.get("GLOBAL", None)
    if mm is None:
        mm = gmm_stream[ws]  # legacy per-watershed cache
    mn, mx = mm['min'], mm['max']
    scale = (mx - mn) if (mx - mn) != 0 else 1.0
    return v * scale + mn

def inverse_flow_transform(x: np.ndarray, kind: str | None) -> np.ndarray:
    if kind == "log1p": return np.expm1(x)
    if kind == "asinh": return np.sinh(x)
    if kind == "sqrt":  return np.square(x)
    return x

def _sanitize(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None: return None
    return torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)

def _freeze_all_bns(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

# =========================
# Collate + statics prep
# =========================
def collate_with_static(batch):
    X = torch.stack([b["X"] for b in batch], 0)
    y = torch.stack([b["y"] for b in batch], 0)
    X = _sanitize(X); y = _sanitize(y)
    meta = [b["meta"] for b in batch]

    def maybe_stack(key):
        if key in batch[0] and isinstance(batch[0][key], torch.Tensor):
            return torch.stack([b[key] for b in batch], 0)
        return None

    DEM  = _sanitize(maybe_stack("DEM"))
    awc  = _sanitize(maybe_stack("awc"))
    fc   = _sanitize(maybe_stack("fc"))
    soil = _sanitize(maybe_stack("soil"))
    return {"X": X, "y": y, "meta": meta, "DEM": DEM, "awc": awc, "fc": fc, "soil": soil}

def _expand_static_to_batch(t: torch.Tensor | None, B: int, channels_expected: int) -> torch.Tensor | None:
    if t is None: return None
    if t.dim() == 2:   t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3: t = t.unsqueeze(0)
    elif t.dim() != 4: raise ValueError(f"Unexpected static tensor dim: {t.dim()}")

    if t.shape[1] != channels_expected:
        if channels_expected == 3 and t.shape[1] == 1: t = t.repeat(1, 3, 1, 1)
        elif channels_expected == 1 and t.shape[1] != 1: t = t[:, :1]
    if t.shape[0] != B: t = t.repeat(B, 1, 1, 1)
    return t

def prep_statics_for_batch(batch, device):
    B = batch["X"].shape[0]
    DEMb  = _expand_static_to_batch(batch.get("DEM",  None), B, 1)
    AWCb  = _expand_static_to_batch(batch.get("awc",  None), B, 1)
    FCb   = _expand_static_to_batch(batch.get("fc",   None), B, 1)
    soilb = _expand_static_to_batch(batch.get("soil", None), B, 3)
    DEMb  = _sanitize(DEMb);  AWCb = _sanitize(AWCb)
    FCb   = _sanitize(FCb);   soilb = _sanitize(soilb)
    if DEMb  is not None: DEMb  = DEMb.to(device, non_blocking=True)
    if AWCb  is not None: AWCb  = AWCb.to(device, non_blocking=True)
    if FCb   is not None: FCb   = FCb.to(device, non_blocking=True)
    if soilb is not None: soilb = soilb.to(device, non_blocking=True)
    return DEMb, AWCb, FCb, soilb

# =========================
# DP wrapper
# =========================
def wrap_dataparallel_if_available(model: nn.Module, device: torch.device, device_ids = None):
    if device.type == "cuda" and torch.cuda.device_count() > 1 and (device_ids is None or len(device_ids) > 1):
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
    return model

# =========================
# Model builder (ResNet-50 + deeper Transformer)
# =========================
def build_model(in_channels: int, device: torch.device,
                fusion_type: str = "film",
                freeze_all_temporal: bool = False,
                unfreeze_last_n: int = 1) -> nn.Module:
    model = ImprovedHydroTransformer(
        in_channels=in_channels,
        # Spatial
        spatial_d_model=32, spatial_pretrained=True, spatial_freeze_stages=0,
        # Temporal
        temporal_d_model=32,
        temporal_heads=2,
        temporal_depth=4,
        temporal_ff_mult=1,
        temporal_dropout=0.1,
        temporal_norm_first=True,
        temporal_use_cls_token=False,
        temporal_checkpoint_path="kleopatra102/solar",
        # Statics/Fusion/Head
        static_d_model=64, fusion_type=fusion_type, output_dim=1,
        map_location=device
    ).to(device)

    if freeze_all_temporal:
        model.freeze_temporal_all(keep_proj=True, keep_norm=True)
        model.unfreeze_temporal_last_n(unfreeze_last_n)

    base = get_base_model(model)
    try:
        _freeze_all_bns(base.spatial_encoder.backbone)
    except Exception:
        pass

    try:
        total = base.total_param_count()
        trainable = base.trainable_param_count()
        print(f"Model params: total={total:,} | trainable={trainable:,}")
    except Exception:
        pass

    return model

def safe_state_dict(m: nn.Module):
    return get_base_model(m).state_dict()

def save_model(model, optimizer, epoch, filename, scaler=None):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    print(f"Saving model at epoch {epoch} -> {filename}")
    payload = {
        'epoch': epoch,
        'model_state_dict': safe_state_dict(model),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if scaler is not None:
        try:
            payload['scaler_state'] = scaler.state_dict()
        except Exception:
            pass
    torch.save(payload, filename)

def load_checkpoint(path, model, optimizer=None, scaler=None, map_location=None, strict=True, load_optim=True):
    ckpt = torch.load(path, map_location=map_location or 'cpu')
    get_base_model(model).load_state_dict(ckpt['model_state_dict'], strict=strict)
    start_epoch = int(ckpt.get('epoch', 0))
    if optimizer is not None and load_optim and 'optimizer_state_dict' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError as e:
            print(f"[warn] Optimizer state incompatible ({e}). Continuing with a fresh optimizer.")
    if scaler is not None and 'scaler_state' in ckpt:
        try: scaler.load_state_dict(ckpt['scaler_state'])
        except Exception: print("[warn] AMP scaler load failed; continuing fresh scaler.")
    print(f"Loaded checkpoint '{os.path.basename(path)}' at epoch {start_epoch}.")
    return start_epoch

# =========================
# Checkpoint-on-interrupt
# =========================
class CheckpointManager:
    def __init__(self, model, optimizer, out_dir="./", scaler=None):
        self.model = model; self.optimizer = optimizer; self.scaler = scaler
        self.out_dir = out_dir; self.current_epoch = 0; self.active = True
    def handler(self, signum, frame):
        if not self.active: return
        fn = os.path.join(self.out_dir, f"model_interrupt_epoch_{self.current_epoch}.pth")
        print(f"\n[Signal {signum}] Interrupt — saving checkpoint: {fn}")
        save_model(self.model, self.optimizer, self.current_epoch, fn, scaler=self.scaler)
        raise KeyboardInterrupt

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, loss_fn, opt, scaler, epoch, device, writer=None, use_amp=True):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}", ncols=100, leave=False)
    for step, batch in enumerate(pbar, start=1):
        X = batch["X"].to(device, non_blocking=True)
        y = batch["y"][:, 0].to(device, non_blocking=True)
        DEMb, AWCb, FCb, soilb = prep_statics_for_batch(batch, device)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=="cuda" and use_amp)):
            y_hat = model(X, DEM=DEMb, awc=AWCb, fc=FCb, soil=soilb)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=-1e6)
            loss = loss_fn(y_hat, y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()

        running += float(loss.detach().item())
        pbar.set_postfix(loss=running/step)

        if writer is not None:
            global_step = (epoch - 1) * len(loader) + step
            writer.add_scalar("train/batch_loss", float(loss.detach().item()), global_step)

    epoch_loss = running / max(1, len(loader))
    if writer is not None:
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
    return epoch_loss

@torch.no_grad()
def eval_loss(model, loader, loss_fn, device, writer=None, epoch=None, split_name="val", use_amp=True):
    model.eval()
    total = 0.0
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        y = batch["y"][:, 0].to(device, non_blocking=True)
        DEMb, AWCb, FCb, soilb = prep_statics_for_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=="cuda" and use_amp)):
            y_hat = model(X, DEM=DEMb, awc=AWCb, fc=FCb, soil=soilb)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=-1e6)
            total += float(loss_fn(y_hat, y).detach().item())
    epoch_loss = total / max(1, len(loader))
    if writer is not None and epoch is not None:
        writer.add_scalar(f"{split_name}/epoch_loss", epoch_loss, epoch)
    return epoch_loss

@torch.no_grad()
def eval_nse_and_plots(model, loader, device, epoch, plots_dir, gmm_stream, flow_transform_kind, use_amp=True):
    model.eval()
    os.makedirs(plots_dir, exist_ok=True)
    all_nse = {}
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        y_norm = batch["y"][:, 0].cpu().numpy()
        DEMb, AWCb, FCb, soilb = prep_statics_for_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=="cuda" and use_amp)):
            yhat_norm = model(X, DEM=DEMb, awc=AWCb, fc=FCb, soil=soilb).detach().cpu().numpy()
        for i, m in enumerate(batch["meta"]):
            ws = m["watershed"]
            y_tr    = _denorm_streamflow(y_norm[i],  ws, gmm_stream)
            yhat_tr = _denorm_streamflow(yhat_norm[i], ws, gmm_stream)
            obs  = inverse_flow_transform(y_tr,    flow_transform_kind)
            pred = inverse_flow_transform(yhat_tr, flow_transform_kind)
            if ws not in all_nse: all_nse[ws] = {"obs": [], "pred": []}
            all_nse[ws]["obs"].append(obs); all_nse[ws]["pred"].append(pred)

    for ws, vals in all_nse.items():
        obs  = np.array(vals["obs"]); pred = np.array(vals["pred"])
        score = nse(obs, pred)
        print(f"[Epoch {epoch}] Watershed {ws} NSE: {score:.3f}")
        logger.info(f"[Epoch {epoch}] Watershed {ws} NSE: {score:.3f}")
        plt.figure(figsize=(5,5))
        plt.scatter(obs, pred, s=10, alpha=0.5)
        mn, mx = obs.min(), obs.max()
        plt.plot([mn, mx], [mn, mx], 'r--', label='1:1')
        plt.xlabel("Observed Flow"); plt.ylabel("Predicted Flow")
        plt.title(f"WS {ws} | NSE={score:.3f} | Epoch {epoch}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"watershed_{ws}_epoch{epoch}.png"), dpi=150)
        plt.close()

# =========================
# Data loaders (split-safe GLOBAL scalers)
# =========================
def _make_dataset(ws_list, fit_scalers_on_this_split=False):
    return WatershedFlowDataset(
        h5_path=H5_PATH,
        csv_path=CSV_PATH,
        static_h5=STATIC_H5,
        variables=None,
        watersheds=ws_list,
        seq_len=SEQ_LEN, stride=1,
        lead_days=LEAD_DAYS,
        start_year=START_YEAR, end_year=END_YEAR,
        drop_nan_targets=True,
        climate_transform_map=CLIMATE_TRANSFORM_MAP,
        streamflow_transform=STREAMFLOW_TRANSFORM,
        min_max_file_climate=MM_CLIM,
        min_max_file_streamflow=MM_FLOW,
        min_max_file_static=MM_STAT,
        min_max_scope="global",
        mm_watersheds=ws_list if fit_scalers_on_this_split else None,  # FIT only on TRAIN
    )

def make_loaders(train_ws, val_ws, test_ws):
    # --- Datasets
    train_ds = _make_dataset(train_ws, fit_scalers_on_this_split=True)   # fits and saves caches
    val_ds   = _make_dataset(val_ws,   fit_scalers_on_this_split=False)  # loads same caches
    test_ds  = _make_dataset(test_ws,  fit_scalers_on_this_split=False)  # loads same caches

    # --- Samplers
    train_sampler = GroupedBatchSampler(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_sampler   = GroupedBatchSampler(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_sampler  = GroupedBatchSampler(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # --- Loaders
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_with_static)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_with_static)
    test_loader  = DataLoader(test_ds,  batch_sampler=test_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_with_static)

    # --- Channels & streamflow scaler dict (GLOBAL)
    C = train_ds[0]["X"].shape[1]
    gmm_stream = train_ds.global_min_max_streamflow
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, C, gmm_stream

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume.")
    parser.add_argument("--no-dp", action="store_true", help="Disable DataParallel.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = USE_AMP_DEF and (not args.no_amp)
    EPOCHS_RUN = int(args.epochs)

    # Hardcoded splits
    print("\n===== Training Split (HARDCODED) =====")
    print(f" Train WS: {TRAIN_WATERSHEDS}")
    print(f"  Val  WS: {VAL_WATERSHEDS}")
    print(f"  Test WS: {TEST_WATERSHEDS}")

    # Loaders + gmm_stream for de-normalization
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, C, gmm_stream = make_loaders(
        TRAIN_WATERSHEDS, VAL_WATERSHEDS, TEST_WATERSHEDS
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=TB_DIR)

    # Model & optim
    model = build_model(
        in_channels=C, device=device,
        fusion_type="film",
        freeze_all_temporal=False
    )
    model = wrap_dataparallel_if_available(model, device, None if args.no_dp else [0,1])

    base = get_base_model(model)
    opt = torch.optim.AdamW(
        base.param_groups(lr_dyn_trunk=1e-5, lr_stat_trunk=1e-5, lr_other=3e-4, weight_decay=1e-4)
    )
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS_RUN, eta_min=1e-5)

    # loss_fn = HydroLossNorm(...)
    loss_fn = nn.MSELoss()

    scaler = torch.amp.GradScaler(enabled=(device.type=="cuda" and USE_AMP))

    # Interrupt-safe checkpoints
    ckpt_mgr = CheckpointManager(model, opt, out_dir=OUT_MODELS_DIR, scaler=scaler)
    signal.signal(signal.SIGINT,  ckpt_mgr.handler)
    signal.signal(signal.SIGTERM, ckpt_mgr.handler)

    # Resume?
    start_epoch = 0
    if args.resume is not None and os.path.isfile(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer=opt, scaler=scaler,
                                      map_location=device, strict=True)
        print(f"Resuming training from epoch {start_epoch+1}...")
    elif args.resume:
        print(f"Warning: checkpoint not found at {args.resume}; starting from scratch.")

    # Train
    best_val = float('inf')
    try:
        for epoch in range(start_epoch + 1, EPOCHS_RUN + 1):
            ckpt_mgr.current_epoch = epoch

            train_loss = train_one_epoch(model, train_loader, loss_fn, opt, scaler, epoch, device,
                                         writer=writer, use_amp=USE_AMP)
            val_loss   = eval_loss(model, val_loader, loss_fn, device,
                                   writer=writer, epoch=epoch, split_name="val", use_amp=USE_AMP)

            scheduler.step()

            # log LR per param group
            if writer is not None:
                for gi, pg in enumerate(opt.param_groups):
                    writer.add_scalar(f"opt/lr_group{gi}", pg["lr"], epoch)

            print(f"[Epoch {epoch}] train={len(train_loader) and train_loss:.4f} | val={len(val_loader) and val_loss:.4f}")
            logger.info(f"[Epoch {epoch}] train={train_loss:.4f} | val={val_loss:.4f}")

            # Save best model
            if val_loss < best_val:
                best_val = val_loss
                save_model(model, opt, epoch,
                           filename=os.path.join(OUT_MODELS_DIR, f"best.pth"),
                           scaler=scaler)

            # Periodic snapshot (distinct filename)
            if epoch % 5 == 0:
                save_model(model, opt, epoch,
                           filename=os.path.join(OUT_MODELS_DIR, f"epoch_{epoch}.pth"),
                           scaler=scaler)

            # Optional: test NSE + plots each epoch
            eval_nse_and_plots(
                model, test_loader, device, epoch,
                plots_dir=OUT_PLOTS_DIR,
                gmm_stream=gmm_stream,
                flow_transform_kind=train_ds.streamflow_transform,
                use_amp=USE_AMP
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted. (Checkpoint saved via signal handler.)")

    # Cleanup
    writer.close()
    del model, opt, scaler, loss_fn, scheduler
    del train_ds, val_ds, test_ds
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

    print(f"\nDone. Best validation loss: {best_val:.4f}")

if __name__ == "__main__":
    main()
