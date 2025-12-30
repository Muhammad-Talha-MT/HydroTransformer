import matplotlib.pyplot as plt
import os
import numpy as np
from data_loader_temporal import WatershedFlowDataset, GroupedBatchSampler, per_ws_collate_optimized
from torch.utils.data import DataLoader


def visualize_day_maps(dataset, watersheds, out_dir="plots_day_maps", n_per_ws=3):
    """
    Visualize and save day-wise maps for each variable in the sequence.
    Each image contains 7 subplots (one for each variable) for each day in the sequence.

    Args:
        dataset: WatershedFlowDataset
        watersheds: list of watershed IDs (ints or strs)
        out_dir: base directory for saving maps
        n_per_ws: number of sequences to plot per watershed
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Saving combined day maps under {out_dir}/")

    for ws in watersheds:
        # indices for this watershed
        indices = [i for i, w in enumerate(dataset.ws_of) if str(ws) in str(w)]
        if not indices:
            print(f"[WARN] No sequences found for watershed {ws}")
            continue

        ws_dir = os.path.join(out_dir, f"ws_{ws}")
        os.makedirs(ws_dir, exist_ok=True)

        for k, idx in enumerate(indices[:n_per_ws]):
            item = dataset[idx]
            X = item["X"].numpy()   # (L,C,H,W), float16
            meta = item["meta"]
            L, C, H, W = X.shape

            # Loop through each timestep in the sequence
            for t in range(L):
                fig, axes = plt.subplots(1, C, figsize=(15, 5))  # Create 1 row and C columns of subplots
                fig.suptitle(f"Watershed {ws} | seq_id={idx} | Day {t+1}", fontsize=16)

                # Loop over variables and plot them
                for c, var in enumerate(meta["variables"]):
                    ax = axes[c]
                    var_data = X[t, c, :, :]  # (H, W)
                    cax = ax.imshow(var_data, cmap='viridis', interpolation='nearest')
                    ax.set_title(var)
                    fig.colorbar(cax, ax=ax, orientation='vertical', label="Normalized value")

                plt.tight_layout()
                plt.subplots_adjust(top=0.85)  # Adjust the top to make space for suptitle

                fname = os.path.join(ws_dir, f"seq{idx}_day{t+1}_combined.png")
                plt.savefig(fname, dpi=150)
                plt.close()
                print(f"[PLOT] Saved {fname}")

if __name__ == "__main__":
    # ---- paths (edit to your machine)
    H5_PATH   = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
    CSV_PATH  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"
    STATIC_H5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

    # ---- split lists (example)
    # train_ws = ALL_WATERSHEDS = [
    # 4096405, 4096515, 4097500, 4097540, 4099000, 4101500, 4101800, 4102500,
    # 4102700, 4104945, 4105000, 4105500, 4105700, 4106000, 4108600, 4108800,
    # 4109000, 4112000, 4112500, 4113000, 4114000, 4115000, 4115265, 4116000,
    # 4117500, 4118500, 4121300, 4121500, 4121944, 4121970, 4122100, 4122200,
    # 4122500, 4124200, 4124500, 4125550, 4126740, 4126970, 4127800, 4142000,
    # 4144500, 4146000, 4146063, 4147500, 4148140, 4148500, 4151500, 4152238,
    # 4154000, 4157005, 4159492, 4159900, 4160600, 4163400, 4164100, 4164300,
    # 4166500, 4167000, 4175600
    # ]  # e.g., ['04127800','04119300', ...]  # Fit scalers on these
    train_ws = ALL_WATERSHEDS = [4121300]  # e.g., ['04142000', ...]

    # ---- config
    variables  = None            # or list like ['prcp','tmin','tmax','srad', ...]
    watersheds = train_ws
    seq_len    = 365
    stride     = 1
    lead_days  = 1

    start_year = 2011
    end_year   = 2011

    climate_transform_map = {
        # Zero-inflated, strictly non-negative:
        'prcp': 'log1p', 'ppt': 'log1p', 'precip': 'log1p',
        # Signed temps: keep identity (commented as reminder)
        # 'tmin': 'identity', 'tmax': 'identity',
        # If you have PET/ET or runoff fields that are non-negative and skewed:
        # 'pet': 'log1p'
    }
    streamflow_transform = "log1p"  # good default for Q

    # Cache files (use the SAME three files for train/val/test; generate them from TRAIN first)
    mm_clim = "../scalers/testing/uplog10__CLIMATE__TRAIN_GLOBAL.json"
    mm_flow = "../scalers/testing/uplog10__FLOW__TRAIN_GLOBAL.json"
    mm_stat = "../scalers/testing/uplog10__STATIC__TRAIN_GLOBAL.json"

    dataset = WatershedFlowDataset(
        h5_path=H5_PATH,
        csv_path=CSV_PATH,
        static_h5=STATIC_H5,
        variables=None,
        watersheds=ALL_WATERSHEDS,
        seq_len=seq_len, stride=1,
        lead_days=1,
        start_year=start_year, end_year=end_year,
        drop_nan_targets=True,
        climate_transform_map={'prcp': 'log1p'},
        streamflow_transform="log1p",
        min_max_file_climate=mm_clim,
        min_max_file_streamflow=mm_flow,
        min_max_file_static=mm_stat,
        min_max_scope="global",
        mm_watersheds=ALL_WATERSHEDS  # fit on train ws
    )

    sampler = GroupedBatchSampler(dataset, batch_size=4, shuffle=True)
    loader  = DataLoader(dataset, batch_sampler=sampler,
                         num_workers=4, pin_memory=True,
                         collate_fn=per_ws_collate_optimized)
    print(f"Total dataset size: {len(dataset)} sequences in {len(sampler)} batches.")

    batch = next(iter(loader))
    X, y = batch["X"].float(), batch["y"].float()
    DEM, awc, fc, soil = batch["DEM"], batch["awc"], batch["fc"], batch["soil"]
    print("Example batch tensors:", X.shape, y.shape)
    def _shape_or_none(t):
        try: return tuple(t.shape)
        except: return None
    print("Static shapes:", _shape_or_none(DEM), _shape_or_none(awc), _shape_or_none(fc), _shape_or_none(soil))
    # Visualize
    visualize_day_maps(dataset, ALL_WATERSHEDS, out_dir="plots/plots_day_maps", n_per_ws=2)