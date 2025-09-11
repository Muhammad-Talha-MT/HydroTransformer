import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import logging
import signal
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn

# ---- your modules
from resnet_timesfm import HydroTransformer  # model with static conditioning
from data_loader import (
    WatershedFlowDataset, GroupedBatchSampler, per_ws_collate
)

# =========================
# Logger
# =========================
def setup_logger(log_file='../logs/training_log_static.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    return logger

logger = setup_logger()

# =========================
# Paths / CUDA
# =========================
h5_path   = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
csv_path  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"
static_h5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# =========================
# Splits
# =========================
train_watersheds = [
    4096405, 4096515, 4097500, 4097540, 4099000, 4101500, 4101800, 4102500,
    4102700, 4104945, 4105000, 4105500, 4105700, 4106000, 4108600, 4108800,
    4109000, 4112000, 4112500, 4113000, 4114000, 4115000, 4115265, 4116000,
    4117500, 4118500, 4121300, 4121500, 4121944, 4121970, 4122100, 4122200,
    4122500, 4124200, 4124500, 4125550, 4126740, 4126970, 4127800, 4142000,
    4144500, 4146000, 4146063, 4147500, 4148140, 4148500, 4151500, 4152238,
    4154000, 4157005, 4159492, 4159900, 4160600, 4163400, 4164100, 4164300,
    4166500, 4167000, 4175600
]
val_watersheds  = [4176000, 4176500]
test_watersheds = [4096405, 4096515]

# =========================
# Datasets / Loaders
# (dataset must return batch keys: "DEM","AWC","FC","soil")
# =========================
common_ds_kwargs = dict(
    h5_path=h5_path,
    csv_path=csv_path,
    static_h5=static_h5,
    variables=None,
    seq_len=365, stride=1,
    lead_days=1,
    start_year=2000, end_year=2001,
    drop_nan_targets=True
)

train_dataset = WatershedFlowDataset(watersheds=train_watersheds, **common_ds_kwargs)
val_dataset   = WatershedFlowDataset(watersheds=val_watersheds,   **common_ds_kwargs)
test_dataset  = WatershedFlowDataset(watersheds=test_watersheds,  **common_ds_kwargs)

train_sampler = GroupedBatchSampler(train_dataset, batch_size=4, shuffle=False)
val_sampler   = GroupedBatchSampler(val_dataset,   batch_size=4, shuffle=False)
test_sampler  = GroupedBatchSampler(test_dataset,  batch_size=4, shuffle=False)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=16, pin_memory=True, collate_fn=per_ws_collate)
val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,   num_workers=16, pin_memory=True, collate_fn=per_ws_collate)
test_loader  = DataLoader(test_dataset,  batch_sampler=test_sampler,  num_workers=16, pin_memory=True, collate_fn=per_ws_collate)


# =========================
# Model
# =========================
C = train_dataset[0]["X"].shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = HydroTransformer(
    in_channels=C,
    d_model=64,           # 128 -> 96
    spatial_depth=2,      # 3 -> 2
    temporal_layers=2,    # 4 -> 2
    n_heads=4,            # 8 -> 4
    add_coords=False,
    ff_mult=2,            # 4x -> 2x
    share_static_encoders=True  # set True for even fewer params
).to(device)
def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
print("Params:", count_trainable(model))
print(model)

# model = torch.nn.DataParallel(model, device_ids=[0, 1])

opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
loss_fn = torch.nn.MSELoss()
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

# =========================
# Helpers
# =========================
def _expand_static_to_batch(t: torch.Tensor | None, B: int, channels_expected: int) -> torch.Tensor | None:
    if t is None:
        return None
    if t.dim() == 2:                    # (H,W)
        t = t.unsqueeze(0).unsqueeze(0) # (1,1,H,W)
    elif t.dim() == 3:                  # (C,H,W) or (1,H,W)
        if t.shape[0] in (1, 3):
            t = t.unsqueeze(0)          # (1,C,H,W)
        else:
            t = t.unsqueeze(0).unsqueeze(0)
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

def _prep_statics_for_batch(batch, device):
    B = batch["X"].shape[0]
    DEM  = batch.get("DEM",  None)
    AWC  = batch.get("AWC",  None)
    FC   = batch.get("FC",   None)
    soil = batch.get("soil", None)
    DEMb  = _expand_static_to_batch(DEM,  B, 1)
    AWCb  = _expand_static_to_batch(AWC,  B, 1)
    FCb   = _expand_static_to_batch(FC,   B, 1)
    soilb = _expand_static_to_batch(soil, B, 3)
    if DEMb  is not None: DEMb  = DEMb.to(device, non_blocking=True)
    if AWCb  is not None: AWCb  = AWCb.to(device, non_blocking=True)
    if FCb   is not None: FCb   = FCb.to(device, non_blocking=True)
    if soilb is not None: soilb = soilb.to(device, non_blocking=True)
    return DEMb, AWCb, FCb, soilb

def safe_state_dict(m: nn.Module):
    return m.module.state_dict() if isinstance(m, nn.DataParallel) else m.state_dict()

def save_model(model, optimizer, epoch, filename="model_checkpoint.pth"):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    print(f"Saving model at epoch {epoch} -> {filename}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': safe_state_dict(model),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)

# =========================
# Interrupt-safe checkpointing
# =========================
class CheckpointManager:
    def __init__(self, model, optimizer, out_dir="./"):
        self.model = model
        self.optimizer = optimizer
        self.out_dir = out_dir
        self.current_epoch = 0

    def handler(self, signum, frame):
        fn = os.path.join(self.out_dir, f"model_interrupt_epoch_{self.current_epoch}.pth")
        print(f"\n[Signal {signum}] Interrupt received — saving checkpoint: {fn}")
        save_model(self.model, self.optimizer, self.current_epoch, fn)
        # Re-raise KeyboardInterrupt-like behavior for clean exit
        raise KeyboardInterrupt

ckpt_mgr = CheckpointManager(model, opt, out_dir=".")
signal.signal(signal.SIGINT,  ckpt_mgr.handler)  # Ctrl+C
signal.signal(signal.SIGTERM, ckpt_mgr.handler)  # kill

# =========================
# NSE + 1:1 plots (after each epoch)
# =========================
def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = obs.astype(np.float64)
    pred = pred.astype(np.float64)
    denom = np.sum((obs - obs.mean())**2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((obs - pred)**2) / denom

def eval_nse_and_plots(model, loader, device, epoch, plots_dir="../plots"):
    model.eval()
    os.makedirs(plots_dir, exist_ok=True)

    all_nse = {}
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y = batch["y"][:, 0].cpu().numpy()  # lead-1
            DEMb, AWCb, FCb, soilb = _prep_statics_for_batch(batch, device)

            y_hat = model(X, DEM=DEMb, AWC=AWCb, FC=FCb, soil=soilb).detach().cpu().numpy()

            # Group by watershed
            for i, m in enumerate(batch["meta"]):
                ws = m["watershed"]
                obs = y[i]
                pred = y_hat[i]
                if ws not in all_nse:
                    all_nse[ws] = {"obs": [], "pred": []}
                all_nse[ws]["obs"].append(obs)
                all_nse[ws]["pred"].append(pred)

    # Compute NSE per watershed and save 1:1 plots
    for ws, vals in all_nse.items():
        obs = np.array(vals["obs"])
        pred = np.array(vals["pred"])
        score = nse(obs, pred)
        print(f"[Epoch {epoch}] Watershed {ws} NSE: {score:.3f}")
        logger.info(f"[Epoch {epoch}] Watershed {ws} NSE: {score:.3f}")

        # 1:1
        plt.figure(figsize=(5, 5))
        plt.scatter(obs, pred, s=10, alpha=0.5)
        mn, mx = obs.min(), obs.max()
        plt.plot([mn, mx], [mn, mx], 'r--', label='1:1')
        plt.xlabel("Observed Flow")
        plt.ylabel("Predicted Flow")
        plt.title(f"Watershed {ws} | NSE={score:.3f} | Epoch {epoch}")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"watershed_{ws}_epoch_{epoch}_1to1.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

# =========================
# Train / Eval / Test
# =========================
def train(model, train_loader, loss_fn, opt, scaler, epoch, device):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100, position=0, leave=True)

    for step, batch in enumerate(progress_bar):
        X = batch["X"].to(device)            # (B,L,C,H,W)
        y = batch["y"][:, 0].to(device)      # (B,)
        DEMb, AWCb, FCb, soilb = _prep_statics_for_batch(batch, device)

        # Log the watershed and its dimensions
        watershed = batch["meta"][0]["watershed"]
        H = batch["meta"][0]["H"]; W = batch["meta"][0]["W"]
        logger.info(f"Training watershed: {watershed}, H: {H}, W: {W}")

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
            y_hat = model(X, DEM=DEMb, AWC=AWCb, FC=FCb, soil=soilb)  # static inputs 👍
            loss = loss_fn(y_hat, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix({"loss": running_loss / (step + 1)})

        torch.cuda.empty_cache()

    return running_loss / len(train_loader)

def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            X = batch["X"].to(device)
            y = batch["y"][:, 0].to(device)
            DEMb, AWCb, FCb, soilb = _prep_statics_for_batch(batch, device)
            y_hat = model(X, DEM=DEMb, AWC=AWCb, FC=FCb, soil=soilb)
            val_loss += loss_fn(y_hat, y).item()

            watershed = batch["meta"][0]["watershed"]
            H = batch["meta"][0]["H"]; W = batch["meta"][0]["W"]
            logger.info(f"Evaluating watershed: {watershed}, H: {H}, W: {W}")
            torch.cuda.empty_cache()

    return val_loss / len(val_loader)

def test(model, test_loader, loss_fn, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            X = batch["X"].to(device)
            y = batch["y"][:, 0].to(device)
            DEMb, AWCb, FCb, soilb = _prep_statics_for_batch(batch, device)

            watershed = batch["meta"][0]["watershed"]
            H = batch["meta"][0]["H"]; W = batch["meta"][0]["W"]
            logger.info(f"Training: 1 - Testing watershed: {watershed}, H: {H}, W: {W}")

            y_hat = model(X, DEM=DEMb, AWC=AWCb, FC=FCb, soil=soilb)
            test_loss += loss_fn(y_hat, y).item()
            torch.cuda.empty_cache()

    return test_loss / len(test_loader)

# =========================
# Train Loop (interrupt-safe) + NSE every epoch
# =========================
num_epochs = 1000
best_val_loss = float('inf')

try:
    for epoch in range(num_epochs):
        ckpt_mgr.current_epoch = epoch + 1  # make epoch visible to the signal handler

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, train_loader, loss_fn, opt, scaler, epoch, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Save model every 50 epochs
        if (epoch + 1) % 5 == 0:
            save_model(model, opt, epoch + 1, filename=f"../models/with_static/model_epoch_{epoch+1}.pth")

        # Validate every 100 epochs
        if (epoch + 1) % 5 == 0:
            val_loss = evaluate(model, val_loader, loss_fn, device)
            print(f"Validation Loss after epoch {epoch + 1}: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, opt, epoch + 1, filename=f"../models/with_static/model_epoch_{epoch+1}.pth")

        # ---- NEW: NSE on test set every epoch + plots
        eval_nse_and_plots(model, test_loader, device, epoch=epoch+1, plots_dir="../plots/with_static/")

except KeyboardInterrupt:
    # Already saved a checkpoint via signal handler; optionally save another with a generic name.
    print("\nTraining interrupted by user. (Checkpoint saved.)")

# =========================
# Final Test
# =========================
print("\nEvaluating final model on test data...")
final_test_loss = test(model, test_loader, loss_fn, device)
print(f"Test Loss: {final_test_loss:.4f}")