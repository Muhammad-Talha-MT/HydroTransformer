#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess climate HDF5 with:
  • prcp/swe: log1p, then standardize using global mean/std from TRAIN years
  • all other dynamic variables: standardize using global mean/std from TRAIN years
  • leap years (T=366) trimmed to 365 BEFORE any transform/stat
Outputs:
  • New HDF5 with transformed+standardized datasets (float32, gzip)
  • JSON with per-variable TRAIN-year mean/std (AFTER log1p for prcp/swe)
"""

import argparse, json, re
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import h5py


# ---------------------- helpers ----------------------

def _trim_last_day_to_365(a: np.ndarray) -> np.ndarray:
    if a.shape[0] == 365:
        return a
    if a.shape[0] == 366:
        return a[:-1]
    raise ValueError(f"Unexpected T={a.shape[0]} (expected 365 or 366).")

def _log1p_inplace(a: np.ndarray) -> np.ndarray:
    # Hydrology-safe: clamp negatives to zero, then log1p
    np.maximum(a, 0.0, out=a)
    np.log1p(a, out=a)
    return a

def _years_from_group(g: h5py.Group) -> List[Tuple[int, str]]:
    r = re.compile(r"^(\d{4})(?:subset\d+)?$")
    out = []
    for k in g.keys():
        m = r.match(k)
        if m:
            out.append((int(m.group(1)), k))
    out.sort(key=lambda x: x[0])
    return out

def _scan_h5(h5_path: str,
             variables: Optional[List[str]] = None,
             watersheds: Optional[List[str]] = None,
             start_year: Optional[int] = None,
             end_year: Optional[int] = None) -> Dict[str, Dict[str, List[Tuple[int, str]]]]:
    """
    Returns {ws: {var: [(year, full_path), ...]}} (years sorted).
    """
    out = {}
    with h5py.File(h5_path, "r") as f:
        groups = [k for k in f.keys() if k.endswith("_watershed")]
        if watersheds:
            ws_set = set(map(str, watersheds))
            groups = [g for g in groups if g.split("_")[0] in ws_set or g in ws_set]
        for ws in groups:
            var_map = {}
            for var in f[ws].keys():
                if not isinstance(f[ws][var], h5py.Group):
                    continue
                if variables and var not in variables:
                    continue
                yrs = _years_from_group(f[ws][var])
                if start_year is not None or end_year is not None:
                    yrs = [(y, k) for (y, k) in yrs
                           if (start_year is None or y >= start_year)
                           and (end_year   is None or y <= end_year)]
                if not yrs:
                    continue
                var_map[var] = [(y, f"{ws}/{var}/{k}") for (y, k) in yrs]
            if var_map:
                out[ws] = var_map
    return out


# ---------------------- stats (TRAIN years) ----------------------

def compute_global_mean_std_all_vars(
    src_h5: str,
    train_years: List[int],
    log_first_vars: List[str],
    variables: Optional[List[str]] = None,
    watersheds: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-variable global mean/std over TRAIN years.
    For vars in log_first_vars, apply log1p BEFORE accumulating stats.
    Leap day is trimmed before any operation.
    """
    struct = _scan_h5(src_h5, variables, watersheds,
                      start_year=min(train_years), end_year=max(train_years))

    sums, sumsqs, counts = {}, {}, {}

    with h5py.File(src_h5, "r") as f:
        for ws, var_map in struct.items():
            for var, pairs in var_map.items():
                for y, path in pairs:
                    if y not in train_years:
                        continue
                    a = f[path][()]  # (T,H,W)
                    a = _trim_last_day_to_365(a).astype(np.float64, copy=False)

                    if var in log_first_vars:
                        # log1p first, then stats
                        np.maximum(a, 0.0, out=a)
                        np.log1p(a, out=a)

                    valid = np.isfinite(a)
                    n = int(valid.sum())
                    if n == 0:
                        continue
                    vals = a[valid]
                    if var not in sums:
                        sums[var] = 0.0
                        sumsqs[var] = 0.0
                        counts[var] = 0
                    sums[var]   += float(vals.sum())
                    sumsqs[var] += float(np.square(vals).sum())
                    counts[var] += n

    stats = {}
    for var in counts.keys():
        n = counts[var]
        if n == 0:
            mu, sd = 0.0, 1.0
        else:
            mu = sums[var] / n
            ex2 = sumsqs[var] / n
            var2 = max(1e-12, ex2 - mu * mu)
            sd = float(np.sqrt(var2))
        stats[var] = {"mean": float(mu), "std": float(sd)}
    return stats


# ---------------------- apply preprocessing ----------------------

def write_preprocessed_climate_all_std(
    src_h5: str,
    dst_h5: str,
    stats: Dict[str, Dict[str, float]],
    log_first_vars: List[str],
    variables: Optional[List[str]] = None,
    watersheds: Optional[List[str]] = None,
    compression: str = "gzip",
    compression_opts: int = 4
):
    """
    Create a new H5 with: (log1p if in log_first_vars) -> standardize using stats[var].
    """
    struct = _scan_h5(src_h5, variables, watersheds)
    with h5py.File(src_h5, "r") as fr, h5py.File(dst_h5, "w", libver="latest") as fw:
        # Copy top-level attrs if you want:
        for k, v in fr.attrs.items():
            fw.attrs[k] = v

        for ws, var_map in struct.items():
            g_ws = fw.create_group(ws)
            # Copy group attrs
            for k, v in fr[ws].attrs.items():
                g_ws.attrs[k] = v

            for var, pairs in var_map.items():
                g_var = g_ws.create_group(var)
                # Copy var-group attrs
                for k, v in fr[f"{ws}/{var}"].attrs.items():
                    g_var.attrs[k] = v

                mu = stats.get(var, {}).get("mean", 0.0)
                sd = stats.get(var, {}).get("std", 1.0)
                if sd <= 0:
                    sd = 1.0

                for y, path in pairs:
                    ds_src = fr[path]
                    a = ds_src[()]                       # (T,H,W)
                    a = _trim_last_day_to_365(a).astype(np.float32, copy=False)

                    # Order: log1p first (for prcp/swe), then standardize
                    if var in log_first_vars:
                        _log1p_inplace(a)

                    a -= mu
                    a *= (1.0 / sd)

                    dset = fw.create_dataset(
                        f"{ws}/{var}/{y}",
                        data=a,
                        dtype="float32",
                        compression=compression,
                        compression_opts=compression_opts,
                        shuffle=True,
                        fletcher32=False,
                    )
                    # Preserve per-dataset attrs
                    for k, v in ds_src.attrs.items():
                        dset.attrs[k] = v


# ---------------------- CLI ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Preprocess climate H5: log1p(prcp/swe) then standardize ALL variables.")
    ap.add_argument("--src-h5", required=True, help="Path to original climate HDF5.")
    ap.add_argument("--dst-h5", required=True, help="Path to write preprocessed HDF5.")
    ap.add_argument("--stats-json", required=True, help="Where to write the TRAIN-year mean/std JSON.")
    ap.add_argument("--train-start-year", type=int, required=True, help="TRAIN start year (inclusive).")
    ap.add_argument("--train-end-year", type=int, required=True, help="TRAIN end year (inclusive).")
    ap.add_argument("--variables", nargs="*", default=None, help="Restrict to these variables (default: all).")
    ap.add_argument("--watersheds", nargs="*", default=None, help="Restrict to these watersheds (names or prefixes).")
    ap.add_argument("--log-first-vars", nargs="*", default=["prcp", "swe"],
                    help="Variables to log1p BEFORE standardization (default: prcp swe).")
    return ap.parse_args()

def main():
    args = parse_args()
    train_years = list(range(args.train_start_year, args.train_end_year + 1))

    # 1) TRAIN-year stats (after log1p for vars in log_first_vars)
    stats = compute_global_mean_std_all_vars(
        src_h5=args.src_h5,
        train_years=train_years,
        log_first_vars=args.log_first_vars,
        variables=args.variables,
        watersheds=args.watersheds,
    )

    Path(args.stats_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_json, "w") as f:
        json.dump({
            "train_years": train_years,
            "log_first_vars": list(args.log_first_vars),
            "stats": stats
        }, f, indent=2)
    print(f"📊 Saved TRAIN-year stats → {args.stats_json}")

    # 2) Write transformed + standardized H5
    write_preprocessed_climate_all_std(
        src_h5=args.src_h5,
        dst_h5=args.dst_h5,
        stats=stats,
        log_first_vars=args.log_first_vars,
        variables=args.variables,
        watersheds=args.watersheds,
    )
    print(f"✅ Saved preprocessed climate H5 → {args.dst_h5}")

if __name__ == "__main__":
    main()


# python preprocess_climate_h5.py \
#   --src-h5 /data/HydroTransformer/daymet/daymet_watersheds_clipped.h5 \
#   --dst-h5 /data/HydroTransformer/daymet/processed_daymet_watersheds_clipped.h5 \
#   --stats-json /data/HydroTransformer/daymet/daymet_std_all_stats_2000_2015.json \
#   --train-start-year 2000 --train-end-year 2013 \
#   --log-first-vars prcp swe \
#   --variables prcp tmin tmax srad vp dayl swe
