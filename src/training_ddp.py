# training_ddp.py
# Clean DDP + graceful interrupt + rank-0-only logging/checkpoints
# Option A: Per-epoch LR schedule, stepping only after optimizer.step() actually ran.

import os
import gc
import signal
import logging
import argparse
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributed.distributed_c10d import ProcessGroupNCCL
from resnet_transformer import ImprovedHydroTransformer  # your model

# ===== Dataset/Sampler (DistributedGroupedBatchSampler must exist in data_loader_temporal.py) =====
from data_loader_temporal import (
    WatershedFlowDataset,
    GroupedBatchSampler,
    DistributedGroupedBatchSampler,
    per_ws_collate_optimized,
)

# =========================
# Paths
# =========================
H5_PATH   = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
CSV_PATH  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"
STATIC_H5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

RUN_TAG        = "temporal_split_allWS_ddp"
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
# Temporal split (defaults)
# =========================
TRAIN_START_YEAR = 2010
TRAIN_END_YEAR   = 2010
TEST_YEAR        = 2012  # example

# =========================
# Cache files
# =========================
MM_CLIM = os.path.join(SCALER_DIR, f"uplog10__CLIMATE__TRAIN_GLOBAL_Y{TRAIN_START_YEAR}_{TRAIN_END_YEAR}.json")
MM_FLOW = os.path.join(SCALER_DIR, f"uplog10__FLOW__TRAIN_GLOBAL_Y{TRAIN_START_YEAR}_{TRAIN_END_YEAR}.json")
MM_STAT = os.path.join(SCALER_DIR, f"uplog10__STATIC__TRAIN_GLOBAL_Y{TRAIN_START_YEAR}_{TRAIN_END_YEAR}.json")

# =========================
# Transforms
# =========================
CLIMATE_TRANSFORM_MAP = {'prcp': 'log1p', 'ppt': 'log1p', 'precip': 'log1p'}
STREAMFLOW_TRANSFORM = "log1p"

# =========================
# Train Config
# =========================
EPOCHS       = 15000
BATCH_SIZE   = 16     # per GPU
NUM_WORKERS  = 4      # per process
PIN_MEMORY   = True
USE_AMP_DEF  = True
GRAD_CLIP    = 1.0

SEQ_LEN    = 5
LEAD_DAYS  = 1

# =========================
# Watersheds (ALL)
# =========================
# ALL_WATERSHEDS = [
#     4096405, 4096515, 4097500, 4097540, 4099000, 4101500, 4101800, 4102500,
#     4102700, 4104945, 4105000, 4105500, 4105700, 4106000, 4108600, 4108800,
#     4109000, 4112000, 4112500, 4113000, 4114000, 4115000, 4115265, 4116000,
#     4117500, 4118500, 4121300, 4121500, 4121944, 4121970, 4122100, 4122200,
#     4122500, 4124200, 4124500, 4125550, 4126740, 4126970, 4127800, 4142000,
#     4144500, 4146000, 4146063, 4147500, 4148140, 4148500, 4151500, 4152238,
#     4154000, 4157005, 4159492, 4159900, 4160600, 4163400, 4164100, 4164300,
#     4166500, 4167000, 4175600, 4176500, 4176000
# ]
ALL_WATERSHEDS = [4157005]

# =========================
# Logger
# =========================
def setup_logger(log_file=LOG_FILE):
    logger = logging.getLogger('TemporalSplitRunDDP')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    return logger

logger = setup_logger()

# =========================
# Dist helpers
# =========================
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main_process() -> bool:
    return get_rank() == 0

def ddp_barrier():
    if is_dist():
        try:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        except TypeError:
            dist.barrier()

# =========================
# Misc utils
# =========================
def get_base_model(m: nn.Module) -> nn.Module:
    if isinstance(m, (nn.DataParallel, DDP)): return m.module
    return m

def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs.astype(np.float64); pred = pred.astype(np.float64)
    denom = np.sum((obs - obs.mean())**2)
    if denom <= 0: return np.nan
    return 1.0 - np.sum((obs - pred)**2) / denom

def _denorm_streamflow(v: np.ndarray, ws: str, gmm_stream: dict) -> np.ndarray:
    mm = gmm_stream.get("GLOBAL", None) or gmm_stream[ws]
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
# Statics prep
# =========================
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
# Model builder
# =========================
def build_model(in_channels: int, device: torch.device,
                fusion_type: str = "film",
                freeze_all_temporal: bool = False,
                unfreeze_last_n: int = 1) -> nn.Module:
    model = ImprovedHydroTransformer(
        in_channels=in_channels,
        spatial_d_model=32, spatial_pretrained=True, spatial_freeze_stages=0,
        temporal_d_model=32, temporal_heads=2, temporal_depth=4,
        temporal_ff_mult=1, temporal_dropout=0.1, temporal_norm_first=True,
        temporal_use_cls_token=False,
        temporal_checkpoint_path="kleopatra102/solar",
        static_d_model=64, fusion_type=fusion_type, output_dim=1,
        map_location=device
    ).to(device)

    if freeze_all_temporal:
        model.freeze_temporal_all(keep_proj=True, keep_norm=True)
        model.unfreeze_temporal_last_n(unfreeze_last_n)

    try: _freeze_all_bns(model.spatial_encoder.backbone)
    except Exception: pass

    try:
        total = model.total_param_count()
        trainable = model.trainable_param_count()
        if is_main_process():
            print(f"Model params: total={total:,} | trainable={trainable:,}")
    except Exception:
        pass

    return model

def safe_state_dict(m: nn.Module):
    return get_base_model(m).state_dict()

def save_model(model, optimizer, epoch, filename, scaler=None):
    if not is_main_process():
        return
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
    if optimizer is not None and load_optim and 'optimizer_state_dict' in ckpt and is_main_process():
        try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError as e: print(f"[warn] Optimizer state incompatible ({e}). Continuing fresh optimizer.")
    if scaler is not None and 'scaler_state' in ckpt:
        try: scaler.load_state_dict(ckpt['scaler_state'])
        except Exception: print("[warn] AMP scaler load failed; continuing fresh scaler.")
    if is_main_process():
        print(f"Loaded checkpoint '{os.path.basename(path)}' at epoch {start_epoch}.")
    return start_epoch

# =========================
# Checkpoint-on-interrupt (all ranks; save on rank-0)
# =========================
class CheckpointManager:
    def __init__(self, model, optimizer, out_dir="./", scaler=None):
        self.model = model; self.optimizer = optimizer; self.scaler = scaler
        self.out_dir = out_dir; self.current_epoch = 0

def install_interrupts(ckpt_mgr: CheckpointManager):
    def _rank():
        return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    def _handler(signum, frame):
        if _rank() == 0:
            fn = os.path.join(ckpt_mgr.out_dir, f"model_interrupt_epoch_{ckpt_mgr.current_epoch}.pth")
            print(f"\n[Signal {signum}] Interrupt — saving checkpoint: {fn}")
            save_model(ckpt_mgr.model, ckpt_mgr.optimizer, ckpt_mgr.current_epoch, fn, scaler=ckpt_mgr.scaler)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT,  _handler)
    signal.signal(signal.SIGTERM, _handler)

@torch.no_grad()
def tb_log_predictions(model, loader, writer, device, epoch, gmm_stream, flow_transform_kind,
                       max_ws: int = 4, max_points: int = 1000, use_amp: bool = True, tag_prefix: str = "split"):
    if loader is None or writer is None or not is_main_process(): return
    model.eval()
    ws_series = {}
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        if not use_amp and X.dtype != torch.float32: X = X.float()
        y_norm = batch["y"][:, 0].cpu().numpy()
        DEMb, AWCb, FCb, soilb = prep_statics_for_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda" and use_amp)):
            yhat_norm = model(X, DEM=DEMb, awc=AWCb, fc=FCb, soil=soilb).detach().cpu().numpy()
        for i, m in enumerate(batch["meta"]):
            ws = m["watershed"]
            t_idx = int(m["start_global"]) + int(m["seq_len"]) - 1
            y_tr    = _denorm_streamflow(y_norm[i],  ws, gmm_stream)
            yhat_tr = _denorm_streamflow(yhat_norm[i], ws, gmm_stream)
            obs  = inverse_flow_transform(y_tr,    flow_transform_kind)
            pred = inverse_flow_transform(yhat_tr, flow_transform_kind)
            s = ws_series.setdefault(ws, {"t": [], "obs": [], "pred": []})
            s["t"].append(t_idx); s["obs"].append(float(obs)); s["pred"].append(float(pred))

    for j, (ws, s) in enumerate(list(ws_series.items())[:max_ws]):
        t   = np.asarray(s["t"]); obs = np.asarray(s["obs"]); pre = np.asarray(s["pred"])
        order = np.argsort(t); t, obs, pre = t[order], obs[order], pre[order]
        if len(t) > max_points:
            sel = np.linspace(0, len(t)-1, max_points).astype(int)
            t, obs, pre = t[sel], obs[sel], pre[sel]
        score = nse(obs, pre)
        writer.add_scalar(f"{tag_prefix}/NSE/{ws}", float(score), epoch)

        fig_ts = plt.figure(figsize=(7, 3))
        plt.plot(t, obs, label="Obs"); plt.plot(t, pre, label="Pred", alpha=0.85)
        plt.xlabel("Day index"); plt.ylabel("Flow"); plt.title(f"WS {ws} — time series")
        plt.legend(); plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/timeseries/{ws}", fig_ts, global_step=epoch)
        plt.close(fig_ts)

        fig_sc = plt.figure(figsize=(4, 4))
        plt.scatter(obs, pre, s=6, alpha=0.6)
        mn, mx = float(np.min(obs)), float(np.max(obs))
        plt.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        plt.xlabel("Obs"); plt.ylabel("Pred"); plt.title(f"WS {ws} — scatter")
        plt.tight_layout()
        writer.add_figure(f"{tag_prefix}/scatter/{ws}", fig_sc, global_step=epoch)
        plt.close(fig_sc)

        res = pre - obs
        writer.add_histogram(f"{tag_prefix}/residuals/{ws}", torch.tensor(res), global_step=epoch)

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, loss_fn, opt, scaler, epoch, device, writer=None, use_amp=True, sampler_with_epoch=None):
    model.train()
    if sampler_with_epoch is not None and hasattr(sampler_with_epoch, "set_epoch"):
        sampler_with_epoch.set_epoch(epoch)

    # Track optimizer.step() occurrence to gate scheduler.step() later
    start_steps = getattr(opt, "_step_count", 0)

    running = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch} [rank {get_rank()}]", ncols=100, leave=False, disable=not is_main_process())
    for step, batch in enumerate(pbar, start=1):
        X = batch["X"].to(device, non_blocking=True)
        if not use_amp and X.dtype != torch.float32: X = X.float()
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
        if is_main_process():
            pbar.set_postfix(loss=running/step)
            if writer is not None:
                global_step = (epoch - 1) * len(loader) + step
                writer.add_scalar("train/batch_loss", float(loss.detach().item()), global_step)

    # avg loss across workers
    epoch_loss_t = torch.tensor([running / max(1, len(loader))], device=device)
    if is_dist(): dist.all_reduce(epoch_loss_t, op=dist.ReduceOp.AVG)
    epoch_loss = float(epoch_loss_t.item())

    made_step = getattr(opt, "_step_count", 0) > start_steps  # NEW: report whether optimizer stepped

    if is_main_process() and writer is not None:
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
    return epoch_loss, made_step

@torch.no_grad()
def eval_loss(model, loader, loss_fn, device, writer=None, epoch=None, split_name="val", use_amp=True):
    if loader is None or not is_main_process(): return None
    model.eval()
    total = 0.0
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        if not use_amp and X.dtype != torch.float32: X = X.float()
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
def eval_nse_and_plots(model, loader, device, epoch, plots_dir, gmm_stream, flow_transform_kind, use_amp=True, tag="test"):
    if loader is None or not is_main_process(): return
    model.eval()
    os.makedirs(plots_dir, exist_ok=True)
    all_nse = {}
    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        if not use_amp and X.dtype != torch.float32: X = X.float()
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
        print(f"[Epoch {epoch}] ({tag}) Watershed {ws} NSE: {score:.3f}")
        logger.info(f"[Epoch {epoch}] ({tag}) Watershed {ws} NSE: {score:.3f}")
        plt.figure(figsize=(5,5))
        plt.scatter(obs, pred, s=10, alpha=0.5)
        mn, mx = obs.min(), obs.max()
        plt.plot([mn, mx], [mn, mx], 'r--', label='1:1')
        plt.xlabel("Observed Flow"); plt.ylabel("Predicted Flow")
        plt.title(f"{tag.upper()} | WS {ws} | NSE={score:.3f} | Epoch {epoch}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{tag}_watershed_{ws}_epoch{epoch}.png"), dpi=150)
        plt.close()

# =========================
# Distributed init/cleanup
# =========================
def setup_distributed():
    # Safer defaults
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")

    launched = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if launched and local_rank >= n:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {n} GPU(s) visible. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
        torch.cuda.set_device(local_rank if launched else 0)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if launched:
        if backend == "nccl":
            # Prefer device-aware PG init if available
            try:
                opts = ProcessGroupNCCL.Options()
                opts.device_id = torch.cuda.current_device()
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=30),
                    pg_options=opts,
                )
            except Exception:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    timeout=timedelta(minutes=30),
                )
        else:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(minutes=30),
            )
    return device, local_rank

def cleanup_distributed():
    if is_dist():
        try:
            ddp_barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

# =========================
# Datasets / Loaders
# =========================
def _make_dataset(ws_list, start_year, end_year, fit_scalers_on_this_split=False):
    return WatershedFlowDataset(
        h5_path=H5_PATH,
        csv_path=CSV_PATH,
        static_h5=STATIC_H5,
        variables=None,
        watersheds=ws_list,
        seq_len=SEQ_LEN, stride=1,
        lead_days=LEAD_DAYS,
        start_year=start_year, end_year=end_year,
        drop_nan_targets=True,
        climate_transform_map=CLIMATE_TRANSFORM_MAP,
        streamflow_transform=STREAMFLOW_TRANSFORM,
        min_max_file_climate=MM_CLIM,
        min_max_file_streamflow=MM_FLOW,
        min_max_file_static=MM_STAT,
        min_max_scope="global",
        mm_watersheds=ws_list if fit_scalers_on_this_split else None,
    )

def prepare_scalers_once(all_ws, train_yr0, train_yr1):
    if is_main_process():
        _ = _make_dataset(all_ws, train_yr0, train_yr1, fit_scalers_on_this_split=True)
        del _
    ddp_barrier()

def make_loaders_temporal(all_ws, train_yr0, train_yr1, test_year, val_year=None, ddp=False, rank=0, world_size=1):
    train_ds = _make_dataset(all_ws, train_yr0, train_yr1, fit_scalers_on_this_split=False)
    val_ds   = _make_dataset(all_ws, val_year, val_year, fit_scalers_on_this_split=False) if val_year else None
    test_ds  = _make_dataset(all_ws, test_year, test_year, fit_scalers_on_this_split=False)

    if ddp:
        train_sampler = DistributedGroupedBatchSampler(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                                                       rank=rank, world_size=world_size, drop_last=True, seed=123)
    else:
        train_sampler = GroupedBatchSampler(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Evaluate/plot only on rank-0
    if is_main_process():
        val_sampler  = GroupedBatchSampler(val_ds,  batch_size=BATCH_SIZE, shuffle=False) if val_ds else None
        test_sampler = GroupedBatchSampler(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_sampler = None; test_sampler = None; val_ds = None

    common_loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
        persistent_workers=False,                 # ← important for clean interrupts
        collate_fn=per_ws_collate_optimized,
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_sampler=train_sampler, **common_loader_kwargs)
    val_loader   = (torch.utils.data.DataLoader(val_ds,   batch_sampler=val_sampler,   **common_loader_kwargs)
                    if val_ds and val_sampler else None)
    test_loader  = (torch.utils.data.DataLoader(test_ds,  batch_sampler=test_sampler,  **common_loader_kwargs)
                    if test_sampler else None)

    C = train_ds[0]["X"].shape[1]
    gmm_stream = train_ds.global_min_max_streamflow
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, C, gmm_stream, train_sampler

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Temporal split training (DDP).")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--val-year", type=int, default=None)
    args = parser.parse_args()

    # Perf toggles
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device, local_rank = setup_distributed()
    print(device, local_rank)
    ddp   = is_dist()
    world = get_world_size()
    rank  = get_rank()
    print(f"DDP: {ddp} | world={world} | rank={rank} | device={device}")
    if is_main_process():
        print(f"DDP: {ddp} | world={world} | rank={rank} | device={device}")
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("torch.cuda.device_count():", torch.cuda.device_count())
        print("\n===== Temporal Split (HARDCODED defaults) =====")
        print(f"  Train Years: {TRAIN_START_YEAR}–{TRAIN_END_YEAR}  | Watersheds: ALL ({len(ALL_WATERSHEDS)})")
        print(f"  Val   Year : {args.val_year if args.val_year else 'None (skipped)'} | Watersheds: ALL")
        print(f"  Test  Year : {TEST_YEAR}                        | Watersheds: ALL")

    USE_AMP    = USE_AMP_DEF and (not args.no_amp)
    EPOCHS_RUN = int(args.epochs)

    # Prepare min/max JSONs once
    prepare_scalers_once(ALL_WATERSHEDS, TRAIN_START_YEAR, TRAIN_END_YEAR)

    # Loaders + global min/max
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, C, gmm_stream, train_sampler = make_loaders_temporal(
        ALL_WATERSHEDS, TRAIN_START_YEAR, TRAIN_END_YEAR, TEST_YEAR,
        val_year=args.val_year, ddp=ddp, rank=rank, world_size=world
    )

    writer = SummaryWriter(log_dir=TB_DIR) if is_main_process() else None

    # Model / Opt
    torch.manual_seed(1234 + rank)
    model = build_model(in_channels=C, device=device, fusion_type="film", freeze_all_temporal=False)

    base_for_groups = get_base_model(model)
    opt = torch.optim.AdamW(
        base_for_groups.param_groups(lr_dyn_trunk=1e-5, lr_stat_trunk=1e-5, lr_other=3e-4, weight_decay=1e-4)
    )
    scheduler = CosineAnnealingLR(opt, T_max=EPOCHS_RUN, eta_min=1e-5)

    # Wrap in DDP (find_unused_parameters needed due to conditional fusion path)
    if ddp:
        model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25,
        )

    loss_fn = nn.MSELoss()
    scaler  = torch.amp.GradScaler(enabled=(device.type=="cuda" and USE_AMP))

    # Interrupt-safe checkpoints (ALL ranks install handler; rank-0 saves)
    ckpt_mgr = CheckpointManager(model, opt, out_dir=OUT_MODELS_DIR, scaler=scaler)
    install_interrupts(ckpt_mgr)

    # Resume
    start_epoch = 0
    if args.resume is not None and os.path.isfile(args.resume):
        start_epoch = load_checkpoint(args.resume, model, map_location=device, strict=True)
        if is_main_process(): print(f"Resuming training from epoch {start_epoch+1}...")
    elif args.resume:
        if is_main_process(): print(f"Warning: checkpoint not found at {args.resume}; starting from scratch.")

    best_key = float('inf')
    best_key_name = "val" if val_loader is not None else "train"

    try:
        for epoch in range(start_epoch + 1, EPOCHS_RUN + 1):
            ckpt_mgr.current_epoch = epoch

            train_loss, made_step = train_one_epoch(
                model, train_loader, loss_fn, opt, scaler, epoch, device,
                writer=writer, use_amp=USE_AMP, sampler_with_epoch=train_sampler
            )

            val_loss = eval_loss(
                model, val_loader, loss_fn, device,
                writer=writer, epoch=epoch, split_name="val", use_amp=USE_AMP
            ) if val_loader else None

            tb_log_predictions(
                model, test_loader, writer, device, epoch,
                gmm_stream=gmm_stream,
                flow_transform_kind=train_ds.streamflow_transform,
                use_amp=USE_AMP,
                max_ws=4, max_points=1500,
                tag_prefix="test2020"
            )

            # ---- Option A: per-epoch scheduler; step only if optimizer stepped
            if made_step:
                scheduler.step()
            elif is_main_process():
                print(f"[Epoch {epoch}] Skipping scheduler.step() (no optimizer.step() this epoch).")

            if is_main_process():
                for gi, pg in enumerate(opt.param_groups):
                    writer.add_scalar(f"opt/lr_group{gi}", pg["lr"], epoch)

                msg = f"[Epoch {epoch}] train={train_loss:.4f}"
                if val_loss is not None: msg += f" | val={val_loss:.4f}"
                print(msg); logger.info(msg)

                track_val = val_loss if val_loss is not None else train_loss
                if track_val < best_key:
                    best_key = track_val
                    save_model(model, opt, epoch,
                               filename=os.path.join(OUT_MODELS_DIR, f"best_{best_key_name}.pth"),
                               scaler=scaler)

                if epoch % 5 == 0:
                    save_model(model, opt, epoch,
                               filename=os.path.join(OUT_MODELS_DIR, f"epoch_{epoch}.pth"),
                               scaler=scaler)

                eval_nse_and_plots(
                    model, test_loader, device, epoch,
                    plots_dir=OUT_PLOTS_DIR,
                    gmm_stream=gmm_stream,
                    flow_transform_kind=train_ds.streamflow_transform,
                    use_amp=USE_AMP,
                    tag="test2020"
                )

            ddp_barrier()

    except KeyboardInterrupt:
        if is_main_process():
            print("\nTraining interrupted.")

    finally:
        if writer is not None:
            try:
                writer.flush()
                writer.close()
            except Exception:
                pass
        cleanup_distributed()

        # free
        del model, opt, scaler, loss_fn, scheduler
        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        if is_main_process():
            print(f"\nDone. Best ({best_key_name}) loss: {best_key:.4f}")

if __name__ == "__main__":
    main()
