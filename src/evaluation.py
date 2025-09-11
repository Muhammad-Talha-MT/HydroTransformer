# evaluate_test_watersheds.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import torch
import matplotlib.pyplot as plt
from resnet_transformer import ImprovedHydroTransformer
from data_loader import (WatershedFlowDataset,
                         GroupedBatchSampler) 
from sklearn.metrics import mean_squared_error
import numpy as np
from training_static_kfold import collate_with_static
# ----------------------
# File paths
# ----------------------
checkpoint_path = "../models/pretrained_resnet18_tformer_MSE_09102025/best.pth"  # trained model checkpoint


# =========================
# Config
# =========================
H5_PATH   = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
CSV_PATH  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"
STATIC_H5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

OUT_MODELS_DIR = "../models/10KF_5SL_2017_2021/kfold_with_static"
OUT_PLOTS_DIR  = "../plots/kfold_with_static"
LOG_FILE       = "../logs/training_kfold_static.log"

os.makedirs(OUT_MODELS_DIR, exist_ok=True)
os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

NUM_FOLDS       = 10
EPOCHS_PER_FOLD = 10
BATCH_SIZE      = 32
NUM_WORKERS     = 4
PIN_MEMORY      = True
LR              = 2e-2
WD              = 1e-2
GRAD_CLIP       = 1.0
USE_AMP         = True

SEQ_LEN    = 5
LEAD_DAYS  = 1
START_YEAR = 2017
END_YEAR   = 2021
# ----------------------
# Test watersheds
# ----------------------
test_watersheds =  [
    4096405, 4096515, 4097500, 4097540, 4099000, 4101500, 4101800, 4102500,
    4102700, 4104945, 4105000, 4105500, 4105700, 4106000, 4108600, 4108800,
    4109000, 4112000, 4112500, 4113000, 4114000, 4115000, 4115265, 4116000,
    4117500, 4118500, 4121300, 4121500, 4121944, 4121970, 4122100, 4122200,
    4122500, 4124200, 4124500, 4125550, 4126740, 4126970, 4127800, 4142000,
    4144500, 4146000, 4146063, 4147500, 4148140, 4148500, 4151500, 4152238,
    4154000, 4157005, 4159492, 4159900, 4160600, 4163400, 4164100, 4164300,
    4166500, 4167000, 4175600
]

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


test_ds  = WatershedFlowDataset(watersheds=test_watersheds,  **common_ds_kwargs)


test_sampler = GroupedBatchSampler(test_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_sampler=test_sampler,
                                          num_workers=4, pin_memory=True, collate_fn=collate_with_static)

# ----------------------
# Load trained model
# ----------------------
C = test_ds[0]["X"].shape[1]
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = ImprovedHydroTransformer(
    in_channels=C,
    # Spatial (as you like)
    spatial_d_model=32, spatial_pretrained=True, spatial_freeze_stages=0,
    # >>> Temporal: must match HF config <<<
    temporal_d_model=32,
    temporal_heads=2,
    temporal_depth=4,
    temporal_ff_mult=1,     # 32 / 64 to match encoder_ffn_dim=32
    temporal_dropout=0.1,
    temporal_norm_first=True,
    temporal_use_cls_token=False,
    temporal_checkpoint_path="kleopatra102/solar",
    # Statics/Fusion/Head
    static_d_model=64, fusion_type='film', output_dim=1,
    map_location=device
).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# ----------------------
# NSE function
# ----------------------
def nse(observed, simulated):
    """
    Nash-Sutcliffe Efficiency
    NSE = 1 - sum((obs-sim)^2)/sum((obs-mean(obs))^2)
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    return 1 - np.sum((observed - simulated) ** 2) / np.sum((observed - np.mean(observed)) ** 2)

# ----------------------
# Evaluation
# ----------------------
all_nse = {}
for batch in test_loader:
    X = batch["X"].to(device)
    y = batch["y"][:, 0].cpu().numpy()  # lead-1
    meta = batch["meta"]

    with torch.no_grad():
        y_hat = model(X).cpu().numpy()

    # Group predictions by watershed
    for i, m in enumerate(meta):
        ws = m["watershed"]
        obs = y[i]
        pred = y_hat[i]
        if ws not in all_nse:
            all_nse[ws] = {"obs": [], "pred": []}
        all_nse[ws]["obs"].append(obs)
        all_nse[ws]["pred"].append(pred)

# ----------------------
# Compute NSE per watershed and plot 1:1
# ----------------------
for ws, vals in all_nse.items():
    obs = np.array(vals["obs"])
    pred = np.array(vals["pred"])
    score = nse(obs, pred)
    print(f"Watershed {ws} NSE: {score:.3f}")

    # 1:1 plot
    plt.figure(figsize=(5,5))
    plt.scatter(obs, pred, s=10, alpha=0.5)
    plt.plot([obs.min(), obs.max()], [obs.min(), obs.max()], 'r--', label='1:1')
    plt.xlabel("Observed Flow")
    plt.ylabel("Predicted Flow")
    plt.title(f"Watershed {ws} | NSE={score:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../plots/pretrained_resnet18_tformer_MSE_09102025/eval_watershed_{ws}.png")
    plt.close()
