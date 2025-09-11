# model_param_summary.py
# ------------------------------------------------------------
# Print a clean table of your model by section with
# total parameters, trainable parameters, and % trainable.
#
# Usage examples:
#   python model_param_summary.py --in-channels 5
#   python model_param_summary.py --in-channels 7 \
#       --temporal-checkpoint-path kashif/time-series-transformer-mv-traffic-hourly \
#       --freeze-all-temporal --unfreeze-last-n 2 --keep-proj --keep-norm
#   python model_param_summary.py --in-channels 5 --use-dp
# ------------------------------------------------------------

import argparse
import torch
import torch.nn as nn

# Import your model file
from resnet_transformer import ImprovedHydroTransformer


# -----------------------------
# Helpers
# -----------------------------
def unwrap_dp(m: nn.Module) -> nn.Module:
    """Unwrap DataParallel for attribute access."""
    return m.module if isinstance(m, nn.DataParallel) else m

def count_params(mod: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in mod.parameters())
    train = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return total, train

def fmt_int(n: int) -> str:
    return f"{n:,}"

def rows_for_model(model: nn.Module):
    """
    Build per-section rows; robust to optional fusion blocks.
    """
    base = unwrap_dp(model)
    rows = []

    def add(name: str, mod: nn.Module | None):
        if mod is None:
            return
        tot, tr = count_params(mod)
        pct = (100.0 * tr / tot) if tot > 0 else 0.0
        rows.append({
            "module": name,
            "params_total": tot,
            "params_trainable": tr,
            "pct_trainable": pct,
        })

    # Major sections
    add("spatial_encoder", base.spatial_encoder)

    # Temporal breakdown
    te = base.temporal_encoder
    add("temporal.proj_in", te.proj_in)
    for i, layer in enumerate(te.encoder.layers):
        add(f"temporal.encoder.layer[{i}]", layer)
    add("temporal.norm", te.norm)

    # Statics + Fusion + Head
    add("static_encoder", base.static_encoder)
    add("fusion.film_gamma", getattr(base, "film_gamma", None))
    add("fusion.film_beta",  getattr(base, "film_beta",  None))
    add("fusion.concat",     getattr(base, "concat",     None))
    add("fusion.cross",      getattr(base, "cross",      None))
    add("output_head", base.output_head)

    # Totals
    tot_all, tr_all = count_params(base)
    rows.append({
        "module": "TOTAL",
        "params_total": tot_all,
        "params_trainable": tr_all,
        "pct_trainable": (100.0 * tr_all / tot_all) if tot_all > 0 else 0.0,
    })
    return rows

def print_rows_pretty(rows):
    # Compute column widths
    colw = {
        "module": max(6, max(len(r["module"]) for r in rows)),
        "total":  max(6, max(len(fmt_int(r["params_total"])) for r in rows)),
        "train":  max(9, max(len(fmt_int(r["params_trainable"])) for r in rows)),
        "pct":    12,
    }
    header = (f"{'Module':<{colw['module']}}  "
              f"{'Params':>{colw['total']}}  "
              f"{'Trainable':>{colw['train']}}  "
              f"{'% Trainable':>{colw['pct']}}")
    line = "-" * len(header)
    print(header)
    print(line)
    for r in rows:
        print(f"{r['module']:<{colw['module']}}  "
              f"{fmt_int(r['params_total']):>{colw['total']}}  "
              f"{fmt_int(r['params_trainable']):>{colw['train']}}  "
              f"{r['pct_trainable']:>{colw['pct']}.1f}%")
    print(line)



def main():
    # args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model exactly like training (but no data needed)
    model = ImprovedHydroTransformer(
        in_channels=7,
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
    # ---- Apply your requested freezing policy (before optional DP wrap) ----
    freeze_all_temporal = False
    unfreeze_last_n = 0
    if freeze_all_temporal:
        model.freeze_temporal_all(keep_proj=True, keep_norm=True)
        model.unfreeze_temporal_last_n(unfreeze_last_n)



    # Show pretrained temporal load status
    loaded = getattr(unwrap_dp(model).temporal_encoder, "pretrained_loaded", False)
    print(f"[Info] temporal_encoder.pretrained_loaded = {loaded}")

    # Produce and print the table (reflects freeze/unfreeze above)
    rows = rows_for_model(model)
    print_rows_pretty(rows)

if __name__ == "__main__":
    main()
