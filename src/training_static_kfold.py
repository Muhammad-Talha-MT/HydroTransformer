# training_static_kfold.py
import os
# 👉 No spaces in the list below
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

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
from torch.utils.tensorboard import SummaryWriter  # ← TensorBoard
from torch.optim.lr_scheduler import CosineAnnealingLR  # ← NEW

from hydro_loss import HydroLossNorm
# ---- your modules
from hydro_transformer_wStatic_pretrained import ImprovedHydroTransformer
# from hydro_vit import HydroViT_PatchTST_Static

from data_loader import (
    WatershedFlowDataset,
    GroupedBatchSampler,
)

# =========================
# Config
# =========================
H5_PATH   = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
CSV_PATH  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"
STATIC_H5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

OUT_MODELS_DIR = "../models/10KF_5SL_2017_2021/"
OUT_PLOTS_DIR  = "../plots/10KF_5SL_2017_2021/"
TB_DIR         = "../runs/10KF_5SL_2017_2021/"   # ← TensorBoard logs
LOG_FILE       = "../logs/10KF_5SL_2017_2021/training_kfold_static.log"

os.makedirs(OUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

NUM_FOLDS       = 10
EPOCHS_PER_FOLD = 15   # ← was 10; train longer
BATCH_SIZE      = 64
NUM_WORKERS     = 4
PIN_MEMORY      = True
LR              = 1e-1
WD              = 1e-2
GRAD_CLIP       = 1.0
USE_AMP         = True

SEQ_LEN    = 5
LEAD_DAYS  = 1
START_YEAR = 2017
END_YEAR   = 2021  # small window for fast k-fold sanity run

# =========================
# Logger
# =========================
def setup_logger(log_file=LOG_FILE):
    logger = logging.getLogger('KFoldStatic')
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
def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs.astype(np.float64)
    pred = pred.astype(np.float64)
    denom = np.sum((obs - obs.mean())**2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((obs - pred)**2) / denom

def _denorm_streamflow(v: np.ndarray, ws: str, gmm_stream: dict) -> np.ndarray:
    """Undo min-max to the (possibly transformed) streamflow scale."""
    mn = gmm_stream[ws]['min']; mx = gmm_stream[ws]['max']
    scale = (mx - mn) if (mx - mn) != 0 else 1.0
    return v * scale + mn

def inverse_flow_transform(x: np.ndarray, kind: str | None) -> np.ndarray:
    """Invert the transform applied before min-max (log1p/asinh/sqrt/identity)."""
    if kind == "log1p":
        return np.expm1(x)
    if kind == "asinh":
        return np.sinh(x)
    if kind == "sqrt":
        return np.square(x)
    return x

def kfold_split(ids, k=5, seed=42):
    ids = np.array(ids, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    folds = np.array_split(ids, k)
    return [fold.tolist() for fold in folds]

# =========================
# Collate + statics prep (matches dataset keys: DEM, awc, fc, soil)
# =========================
def collate_with_static(batch):
    X = torch.stack([b["X"] for b in batch], 0)
    y = torch.stack([b["y"] for b in batch], 0)
    meta = [b["meta"] for b in batch]

    def maybe_stack(key):
        if key in batch[0] and isinstance(batch[0][key], torch.Tensor):
            return torch.stack([b[key] for b in batch], 0)
        return None

    DEM  = maybe_stack("DEM")
    awc  = maybe_stack("awc")   # lower-case
    fc   = maybe_stack("fc")    # lower-case
    soil = maybe_stack("soil")
    return {"X": X, "y": y, "meta": meta, "DEM": DEM, "awc": awc, "fc": fc, "soil": soil}

def _expand_static_to_batch(t: torch.Tensor | None, B: int, channels_expected: int) -> torch.Tensor | None:
    if t is None: return None
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif t.dim() == 3:
        t = t.unsqueeze(0)               # (1,C,H,W)
    elif t.dim() == 4:
        pass
    else:
        raise ValueError(f"Unexpected static tensor dim: {t.dim()}")

    if t.shape[1] != channels_expected:
        if channels_expected == 3 and t.shape[1] == 1:
            t = t.repeat(1, 3, 1, 1)
        elif channels_expected == 1 and t.shape[1] != 1:
            t = t[:, :1]

    if t.shape[0] != B:
        t = t.repeat(B, 1, 1, 1)
    return t

def prep_statics_for_batch(batch, device):
    B = batch["X"].shape[0]
    DEMb  = _expand_static_to_batch(batch.get("DEM",  None), B, 1)
    AWCb  = _expand_static_to_batch(batch.get("awc",  None), B, 1)  # lower-case
    FCb   = _expand_static_to_batch(batch.get("fc",   None), B, 1)  # lower-case
    soilb = _expand_static_to_batch(batch.get("soil", None), B, 3)
    if DEMb  is not None: DEMb  = DEMb.to(device, non_blocking=True)
    if AWCb  is not None: AWCb  = AWCb.to(device, non_blocking=True)
    if FCb   is not None: FCb   = FCb.to(device, non_blocking=True)
    if soilb is not None: soilb = soilb.to(device, non_blocking=True)
    return DEMb, AWCb, FCb, soilb

# =========================
# Convenience: wrap DP + make optimizer
# =========================
def wrap_dataparallel_if_available(model: nn.Module, device: torch.device, device_ids = None):
    """Optional helper to wrap in DataParallel."""
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
    return model


def make_optimizer(
    model: nn.Module,
    weight_decay: float = 1e-2,
    lr_pretrained: float = 1e-4,
    lr_new: float = 5e-4,
) -> torch.optim.Optimizer:
    """
    Two-group optimizer:
      - pretrained transformer blocks (temporal encoder layers) -> lr_pretrained
      - small adapters & new modules (proj_in, norm, spatial, statics, fusion, head) -> lr_new
    """
    base = get_base_model(model)

    # collect params
    temporal_blocks = []  # encoder layers that are (partially) trainable
    temporal_small  = []  # proj_in + norm
    other_modules   = []  # spatial/static/fusion/head

    # temporal blocks (only those with requires_grad=True)
    for layer in base.temporal_encoder.encoder.layers:
        if any(p.requires_grad for p in layer.parameters()):
            temporal_blocks += [p for p in layer.parameters() if p.requires_grad]

    # small temporal adapters
    temporal_small += [p for p in base.temporal_encoder.proj_in.parameters() if p.requires_grad]
    temporal_small += [p for p in base.temporal_encoder.norm.parameters()    if p.requires_grad]

    # everything else
    for mod in [base.spatial_encoder, base.static_encoder, getattr(base, "film_gamma", None),
                getattr(base, "film_beta", None), getattr(base, "concat", None),
                getattr(base, "cross", None), base.output_head]:
        if mod is None:
            continue
        other_modules += [p for p in mod.parameters() if p.requires_grad]

    param_groups = []
    if len(temporal_blocks) > 0:
        param_groups.append({"params": temporal_blocks, "lr": lr_pretrained, "weight_decay": weight_decay})
    small_and_new = list(temporal_small) + list(other_modules)
    if len(small_and_new) > 0:
        param_groups.append({"params": small_and_new, "lr": lr_new, "weight_decay": weight_decay})

    # fallback: if nothing matched (shouldn't happen), train all trainable
    if not param_groups:
        param_groups.append({"params": filter(lambda p: p.requires_grad, base.parameters()),
                             "lr": lr_new, "weight_decay": weight_decay})

    opt = torch.optim.AdamW(param_groups)
    return opt

    
# =========================
# Model builder + DataParallel
# =========================

def build_model(
    in_channels: int,
    device: torch.device,
    fusion_type: str = "film",
    freeze_all_temporal: bool = True,
    unfreeze_last_n: int = 1,
) -> nn.Module:
    """
    Build the improved HydroTransformer model.
    """
    model = ImprovedHydroTransformer(
        in_channels=in_channels,
        spatial_d_model=64,          # == d_model of HF encoder
        spatial_depth=3,
        temporal_d_model=64,         # == d_model
        temporal_heads=2,            # == encoder_attention_heads
        temporal_depth=2,            # == encoder_layers
        temporal_ff_mult=0.5,        # int(64*0.5)=32 == encoder_ffn_dim
        temporal_dropout=0.1,
        temporal_norm_first=True,
        temporal_checkpoint_path="kashif/time-series-transformer-mv-traffic-hourly",
        static_d_model=64,
        fusion_type=fusion_type,
        output_dim=1,
        head_dropout=0.1,
    ).to(device)

    base = get_base_model(model)

    # Freeze only if pretrained actually loaded; then unfreeze last 2 layers
    if freeze_all_temporal and getattr(base.temporal_encoder, "pretrained_loaded", False):
        base.freeze_temporal_all(keep_proj=False, keep_norm=False)
        if unfreeze_last_n and unfreeze_last_n > 0:
            base.unfreeze_temporal_last_n(unfreeze_last_n)
        print("[Build] Pretrained temporal weights found → training last 2 temporal layers.")
    else:
        print("[Build] No compatible temporal pretrained weights → leaving all temporal layers trainable.")

    # report
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={total:,} | trainable={trainable:,}")

    return model


def get_base_model(m: nn.Module) -> nn.Module:
    """Return the underlying model (unwrap DataParallel if present)."""
    return m.module if isinstance(m, nn.DataParallel) else m

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

def load_checkpoint(path, model, optimizer=None, scaler=None, map_location=None, strict=True):
    ckpt = torch.load(path, map_location=map_location or 'cpu')
    # Load into underlying module to be agnostic to DP/non-DP
    get_base_model(model).load_state_dict(ckpt['model_state_dict'], strict=strict)
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None and 'scaler_state' in ckpt:
        try:
            scaler.load_state_dict(ckpt['scaler_state'])
        except Exception:
            print("Warning: failed to load scaler_state; continuing without AMP scaler resume.")
    start_epoch = int(ckpt.get('epoch', 0))
    print(f"Loaded checkpoint '{os.path.basename(path)}' at epoch {start_epoch}.")
    return start_epoch

# =========================
# Interrupt-safe checkpointing
# =========================
class CheckpointManager:
    def __init__(self, model, optimizer, out_dir="./", scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.out_dir = out_dir
        self.current_epoch = 0
        self.active = True
    def handler(self, signum, frame):
        if not self.active: return
        fn = os.path.join(self.out_dir, f"model_interrupt_epoch_{self.current_epoch}.pth")
        print(f"\n[Signal {signum}] Interrupt received — saving checkpoint: {fn}")
        save_model(self.model, self.optimizer, self.current_epoch, fn, scaler=self.scaler)
        raise KeyboardInterrupt

# =========================
# Train / Eval (+ TensorBoard logging)
# =========================
def train_one_epoch(model, loader, loss_fn, opt, scaler, epoch, device, writer=None, fold_idx=0):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}", ncols=100, leave=False)
    for step, batch in enumerate(pbar, start=1):
        X = batch["X"].to(device)
        y = batch["y"][:, 0].to(device)
        DEMb, AWCb, FCb, soilb = prep_statics_for_batch(batch, device)

        # (optional) log details — consider reducing to per-epoch to cut I/O:
        ws = batch["meta"][0]["watershed"]; H = batch["meta"][0]["H"]; W = batch["meta"][0]["W"]
        logger.info(f"Training watershed: {ws}, H:{H}, W:{W}")

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=="cuda" and USE_AMP)):
            y_hat = model(X, DEM=DEMb, awc=AWCb, fc=FCb, soil=soilb)
            loss = loss_fn(y_hat, y)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()

        running += float(loss.detach().item())
        avg = running / step
        pbar.set_postfix(loss=avg)

        # TensorBoard per-batch (optional; comment out if too chatty)
        if writer is not None:
            global_step = (epoch - 1) * len(loader) + step
            writer.add_scalar(f"fold_{fold_idx}/train_batch_loss", float(loss.detach().item()), global_step)

    epoch_loss = running / max(1, len(loader))
    if writer is not None:
        writer.add_scalar(f"fold_{fold_idx}/train_epoch_loss", epoch_loss, epoch)
    return epoch_loss

@torch.no_grad()
def eval_loss(model, loader, loss_fn, device, writer=None, epoch=None, fold_idx=0):
    model.eval()
    total = 0.0
    for batch in loader:
        X = batch["X"].to(device)
        y = batch["y"][:, 0].to(device)
        DEMb, AWCb, FCb, soilb = prep_statics_for_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type=="cuda" and USE_AMP)):
            y_hat = model(X, DEM=DEMb, awc=AWCb, fc=FCb, soil=soilb)
            total += float(loss_fn(y_hat, y).detach().item())
    epoch_loss = total / max(1, len(loader))
    if writer is not None and epoch is not None:
        writer.add_scalar(f"fold_{fold_idx}/val_epoch_loss", epoch_loss, epoch)
    return epoch_loss

@torch.no_grad()
def eval_nse_and_plots(model, loader, device, epoch, plots_dir, gmm_stream, flow_transform_kind):
    """
    Compute NSE on the ORIGINAL flow scale by denormalizing and inverting transform.
    """
    model.eval()
    os.makedirs(plots_dir, exist_ok=True)
    all_nse = {}

    for batch in loader:
        X = batch["X"].to(device)
        y_norm = batch["y"][:, 0].cpu().numpy()  # normalized targets
        DEMb, AWCb, FCb, soilb = prep_statics_for_batch(batch, device)
        yhat_norm = model(X, DEM=DEMb, awc=AWCb, fc=FCb, soil=soilb).detach().cpu().numpy()

        for i, m in enumerate(batch["meta"]):
            ws = m["watershed"]  # e.g., "4176000_watershed"
            # First undo min-max (back to transformed domain), then invert transform
            y_tr    = _denorm_streamflow(y_norm[i],  ws, gmm_stream)
            yhat_tr = _denorm_streamflow(yhat_norm[i], ws, gmm_stream)

            obs  = inverse_flow_transform(y_tr,    flow_transform_kind)
            pred = inverse_flow_transform(yhat_tr, flow_transform_kind)

            if ws not in all_nse:
                all_nse[ws] = {"obs": [], "pred": []}
            all_nse[ws]["obs"].append(obs)
            all_nse[ws]["pred"].append(pred)

    # Per-watershed NSE + plots
    for ws, vals in all_nse.items():
        obs  = np.array(vals["obs"])
        pred = np.array(vals["pred"])
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
        plt.savefig(os.path.join(plots_dir, f"watershed_{ws}_epoch_{epoch}_1to1.png"), dpi=150)
        plt.close()

# =========================
# Build loaders per split
# =========================
def make_loaders(train_ws, val_ws, test_ws):
    common_ds_kwargs = dict(
        h5_path=H5_PATH,
        csv_path=CSV_PATH,
        static_h5=STATIC_H5,
        variables=None,
        seq_len=SEQ_LEN, stride=1,
        lead_days=LEAD_DAYS,
        start_year=START_YEAR, end_year=END_YEAR,
        drop_nan_targets=True
    )
    train_ds = WatershedFlowDataset(watersheds=train_ws, **common_ds_kwargs)
    val_ds   = WatershedFlowDataset(watersheds=val_ws,   **common_ds_kwargs)
    test_ds  = WatershedFlowDataset(watersheds=test_ws,  **common_ds_kwargs)

    # ← shuffle True for training
    train_sampler = GroupedBatchSampler(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_sampler   = GroupedBatchSampler(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_sampler  = GroupedBatchSampler(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_with_static)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_with_static)
    test_loader  = DataLoader(test_ds,  batch_sampler=test_sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                              collate_fn=collate_with_static)

    C = train_ds[0]["X"].shape[1]
    gmm_stream = train_ds.global_min_max_streamflow  # for de-normalization
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, C, gmm_stream

# =========================
# Watersheds / K-Fold splits (example small pool)
# =========================
POOL_WATERSHEDS = [
    4096405, 4096515, 4097500, 4097540, 4099000, 4101500, 4101800, 4102500,
    4102700, 4104945, 4105000, 4105500, 4105700, 4106000, 4108600, 4108800,
    4109000, 4112000, 4112500, 4113000, 4114000, 4115000, 4115265, 4116000,
    4117500, 4118500, 4121300, 4121500, 4121944, 4121970, 4122100, 4122200,
    4122500, 4124200, 4124500, 4125550, 4126740, 4126970, 4127800, 4142000,
    4144500, 4146000, 4146063, 4147500, 4148140, 4148500, 4151500, 4152238,
    4154000, 4157005, 4159492, 4159900, 4160600, 4163400, 4164100, 4164300,
    4166500, 4167000, 4175600, 4176000, 4176500
]
TEST_WATERSHEDS = [4176000, 4176500]  # fixed hold-out
POOL_WATERSHEDS = [w for w in POOL_WATERSHEDS if w not in TEST_WATERSHEDS]
FOLD_SPLITS = kfold_split(POOL_WATERSHEDS, k=NUM_FOLDS, seed=42)

# =========================
# Main k-fold loop
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pth to resume training from.")
    parser.add_argument("--no-dp", action="store_true", help="Disable DataParallel.")
    args = parser.parse_args()

    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_overall = float('inf')

    for fold_idx, val_ws in enumerate(FOLD_SPLITS, start=1):
        print(f"\n========== Fold {fold_idx}/{NUM_FOLDS} ==========")
        train_ws = [w for w in POOL_WATERSHEDS if w not in val_ws]
        test_ws  = TEST_WATERSHEDS[:]  # fixed
        print(f"Train WS: {train_ws}")
        print(f" Val WS: {val_ws}")    
        print(f"Test WS: {test_ws}")

        # Loaders + gmm_stream for de-normalization
        train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, C, gmm_stream = make_loaders(train_ws, val_ws, test_ws)

        # TensorBoard writer (per-fold)
        tb_fold_dir = os.path.join(TB_DIR, f"fold_{fold_idx}")
        writer = SummaryWriter(log_dir=tb_fold_dir)

        # Model & optim
        model = build_model(
            in_channels=C,
            device=device,
            fusion_type="film",
            freeze_all_temporal=True,     # freeze if pretrained exists
            unfreeze_last_n=1,            # train last 2 encoder layers
        )
        model = wrap_dataparallel_if_available(model, device, None if args.no_dp else [0,1,2,3])

        # Lighter WD and slightly higher LR for pretrained layers
        opt = make_optimizer(
            model,
            weight_decay=1e-4,   # ← was 1e-2
            lr_pretrained=3e-4,  # ← a bit higher for the last-2 layers
            lr_new=1e-3,         # ← head/adapters
        )

        # Cosine schedule (epoch-based)
        eta_min = 1e-5
        scheduler = CosineAnnealingLR(opt, T_max=EPOCHS_PER_FOLD, eta_min=eta_min)

        loss_fn = HydroLossNorm(
            w_huber=0.8, w_nse=0.15, w_kge=0.05,
            huber_beta=0.1,
            sst_floor_per_sample=1e-3,
            nse_clip=50.0,
            squash="sigmoid",   # ← smoother gradients than clamp
        )

        scaler  = torch.amp.GradScaler(enabled=(device.type=="cuda" and USE_AMP))

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

        best_val = float('inf')
        try:
            for epoch in range(start_epoch + 1, EPOCHS_PER_FOLD + 1):
                ckpt_mgr.current_epoch = epoch

                train_loss = train_one_epoch(model, train_loader, loss_fn, opt, scaler, epoch, device,
                                             writer=writer, fold_idx=fold_idx)
                val_loss   = eval_loss(model, val_loader, loss_fn, device,
                                       writer=writer, epoch=epoch, fold_idx=fold_idx)

                # Scheduler step (epoch)
                scheduler.step()

                print(f"[Fold {fold_idx}] Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")
                logger.info(f"[Fold {fold_idx}] Epoch {epoch} | train={train_loss:.4f} | val={val_loss:.4f}")

                # Save best in this fold
                if val_loss < best_val:
                    best_val = val_loss
                    save_model(model, opt, epoch, filename=os.path.join(
                        OUT_MODELS_DIR, f"best_fold{fold_idx}.pth"), scaler=scaler)

                # Periodic snapshot
                if epoch % 5 == 0:
                    save_model(model, opt, epoch, filename=os.path.join(
                        OUT_MODELS_DIR, f"fold{fold_idx}.pth"), scaler=scaler)

                # Per-epoch NSE on fixed test set (DENORMALIZED + inverse transform)
                eval_nse_and_plots(
                    model, test_loader, device, epoch,
                    plots_dir=os.path.join(OUT_PLOTS_DIR, f"fold_{fold_idx}"),
                    gmm_stream=gmm_stream,
                    flow_transform_kind=train_ds.streamflow_transform
                )

            best_overall = min(best_overall, best_val)

        except KeyboardInterrupt:
            print("\nTraining interrupted. (Checkpoint saved via signal handler.)")

        # Cleanup this fold
        writer.close()  # close TB writer
        del model, opt, scaler, loss_fn, scheduler
        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

    print(f"\nK-Fold complete. Best validation loss across folds: {best_overall:.4f}")

if __name__ == "__main__":
    main()
