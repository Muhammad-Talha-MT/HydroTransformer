#!/usr/bin/env python3
# h5_to_csv.py
# Export per-variable daily means from climate/dynamic HDF5 to CSVs.
# - Rows: dates (one per day, per year; leap handling configurable)
# - Cols: watershed IDs (prefix before "_watershed")
# Robust to year nodes being either Groups (containing datasets) or Datasets directly.

import os
import re
import h5py
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Dict, List

YEAR_RE = re.compile(r"^(\d{4})(?:subset\d+)?$")


# ----------------------------
# HDF5 helpers
# ----------------------------
def _pick_first_dataset(node):
    """Return a dataset whether `node` is already a Dataset or a Group containing datasets."""
    if isinstance(node, h5py.Dataset):
        return node
    for k in node.keys():
        obj = node[k]
        if isinstance(obj, h5py.Dataset):
            return obj
    return None


def _iter_watershed_groups(f: h5py.File, watersheds: Optional[Sequence[str]] = None) -> List[str]:
    """
    Return watershed group names to process. Pattern: '<id>_watershed'.
    If `watersheds` is provided, items can be either exact group names or id prefixes.
    """
    groups = [k for k in f.keys() if k.endswith("_watershed")]
    if not watersheds:
        return groups
    ws_set = set(map(str, watersheds))
    out = []
    for g in groups:
        ws_id = g.split("_")[0]
        if g in ws_set or ws_id in ws_set:
            out.append(g)
    return out


def _available_year_nodes(var_grp: h5py.Group, start_year: Optional[int], end_year: Optional[int]) -> List[str]:
    years = []
    for ky in var_grp.keys():
        m = YEAR_RE.match(ky)
        if not m:
            continue
        y = int(m.group(1))
        if (start_year is not None and y < start_year) or (end_year is not None and y > end_year):
            continue
        years.append((y, ky))
    years.sort(key=lambda x: x[0])
    return [ky for _, ky in years]


# ----------------------------
# Date helpers
# ----------------------------
def _dates_starting_jan1(year: int, length: int) -> pd.DatetimeIndex:
    """Produce `length` consecutive dates starting at Jan 1 of the given year."""
    return pd.date_range(f"{year}-01-01", periods=length, freq="D")


# ----------------------------
# Transforms (optional)
# ----------------------------
def _transform_inplace(arr: np.ndarray, kind: Optional[str]):
    if kind in (None, "identity"):
        return arr
    if kind == "log1p":
        np.maximum(arr, 0.0, out=arr)
        np.log1p(arr, out=arr)
    elif kind == "asinh":
        np.arcsinh(arr, out=arr)
    elif kind == "sqrt":
        np.maximum(arr, 0.0, out=arr)
        np.sqrt(arr, out=arr)
    else:
        raise ValueError(f"Unsupported transform: {kind}")
    return arr


# ----------------------------
# Core export (one variable)
# ----------------------------
def export_variable_timeseries_csv(
    h5_path: str,
    variable: str,
    output_csv: str,
    watersheds: Optional[Sequence[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    leap_strategy: str = "drop_last",  # "drop_last" (trim to 365 by dropping last day) or "drop_feb29"
    transform_map: Optional[Dict[str, str]] = None,  # e.g., {"prcp": "log1p"}
    dtype_out: str = "float32",
    union_of_years: bool = True,  # True: union of dates across watersheds; False: intersection
) -> None:
    """
    Writes one CSV for `variable`.
      - Rows: dates
      - Cols: watershed IDs (prefix before '_watershed')
    Missing days for a watershed are NaN (when union_of_years=True).
    """
    transform_map = transform_map or {}
    v_transform = transform_map.get(variable, None)

    per_ws_series: Dict[str, pd.Series] = {}
    all_dates = None  # track union/intersection

    with h5py.File(h5_path, "r") as f:
        ws_groups = _iter_watershed_groups(f, watersheds)

        for ws_grp in ws_groups:
            if variable not in f[ws_grp]:
                # Skip watersheds that don't have this variable
                continue

            var_grp = f[ws_grp][variable]
            year_nodes = _available_year_nodes(var_grp, start_year, end_year)
            if not year_nodes:
                continue

            daily_vals = []
            daily_dates = []

            for yn in year_nodes:
                node = var_grp[yn]
                ds = _pick_first_dataset(node)
                if ds is None:
                    # No dataset under this year node
                    continue

                a = ds[()]  # Expect (T, H, W)
                if a.ndim != 3:
                    raise ValueError(f"{ws_grp}/{variable}/{yn} expected (T,H,W), got {a.shape}")
                T, H, W = a.shape

                # Sanitize
                a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

                # Optional transform BEFORE spatial averaging
                _transform_inplace(a, v_transform)

                # Leap handling:
                m = YEAR_RE.match(yn)
                assert m is not None, f"Year format unexpected: {yn}"
                year = int(m.group(1))

                if T == 366 and leap_strategy == "drop_last":
                    # Mirror your dataset loader's behavior: trim last day -> length 365
                    a = a[:-1]
                    T = 365
                elif T == 366 and leap_strategy == "drop_feb29":
                    # Build full calendar to locate Feb 29, then drop it from `a`
                    full_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
                    feb29_mask = ~((full_dates.month == 2) & (full_dates.day == 29))
                    a = a[feb29_mask]  # shape becomes (365, H, W)
                    T = 365
                elif T not in (365, 366):
                    raise ValueError(f"{ws_grp}/{variable}/{yn} has T={T}, expected 365 or 366")

                # Spatial mean per day
                day_mean = a.mean(axis=(1, 2)).astype(np.float32, copy=False)  # shape (T_after_trim,)

                # Dates aligned to the (possibly trimmed) length
                dates = _dates_starting_jan1(year, len(day_mean))

                daily_vals.append(day_mean)
                daily_dates.append(dates)

            if not daily_vals:
                continue

            ws_id = ws_grp.split("_")[0]
            vals = np.concatenate(daily_vals, axis=0)
            idx = pd.DatetimeIndex(np.concatenate(daily_dates, axis=0))
            s = pd.Series(vals.astype(dtype_out), index=idx, name=ws_id)
            per_ws_series[ws_id] = s

            # Track global date index
            if all_dates is None:
                all_dates = set(s.index)
            else:
                if union_of_years:
                    all_dates |= set(s.index)
                else:
                    all_dates &= set(s.index)

    if not per_ws_series:
        raise RuntimeError(f"No data found for variable='{variable}' in the given filters.")

    # Align into a single DataFrame
    all_dates_idx = pd.DatetimeIndex(sorted(all_dates))
    df = pd.DataFrame(index=all_dates_idx)
    for ws_id, s in per_ws_series.items():
        df[ws_id] = s.reindex(all_dates_idx)

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.index.name = "date"
    df.to_csv(output_csv, float_format="%.6f")
    print(f"[OK] Wrote {output_csv}  (rows={len(df)}, cols={len(df.columns)})")


# ----------------------------
# Bulk export (all or some variables)
# ----------------------------
def export_all_variables(
    h5_path: str,
    output_dir: str,
    variables: Optional[Sequence[str]] = None,   # None -> discover all variables present
    watersheds: Optional[Sequence[str]] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    leap_strategy: str = "drop_last",
    transform_map: Optional[Dict[str, str]] = None,
    union_of_years: bool = True,
    dtype_out: str = "float32",
) -> None:
    # Discover variables if not specified
    if variables is None:
        with h5py.File(h5_path, "r") as f:
            ws_groups = _iter_watershed_groups(f, watersheds)
            found = set()
            for g in ws_groups:
                for var in f[g].keys():
                    if isinstance(f[g][var], h5py.Group):
                        found.add(var)
            variables = sorted(found)

    if not variables:
        raise RuntimeError("No variables to export (none found and none specified).")

    os.makedirs(output_dir, exist_ok=True)
    for var in variables:
        out_csv = os.path.join(output_dir, f"{var}_daily_mean.csv")
        export_variable_timeseries_csv(
            h5_path=h5_path,
            variable=var,
            output_csv=out_csv,
            watersheds=watersheds,
            start_year=start_year,
            end_year=end_year,
            leap_strategy=leap_strategy,
            transform_map=transform_map,
            union_of_years=union_of_years,
            dtype_out=dtype_out,
        )

if __name__ == "__main__":
    # Example usage:
    # python export_dynamic_means_to_csv.py
    H5 = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
    OUTDIR = "../../data/processed/daymet_csv"
    # Choose variables explicitly or leave as None to export all discovered
    VARS = ["prcp", "tmin", "tmax", "dayl", "swe", "vp", "srad"]  # or None

    # Optional per-variable transforms (same semantics as your loader)
    TRANSFORMS = {
        # "precip": "log1p",
        # "tmin": "identity",
        # "tmax": "identity",
    }

    export_all_variables(
        h5_path=H5,
        output_dir=OUTDIR,
        variables=VARS,                 # or None
        watersheds=None,                # or ["01013500", "03339000_watershed"]
        start_year=None,                # e.g., 2000
        end_year=None,                  # e.g., 2015
        leap_strategy="drop_last",      # to mirror your dataset behavior
        transform_map=TRANSFORMS,
        union_of_years=True,            # align on union of dates across watersheds
    )
