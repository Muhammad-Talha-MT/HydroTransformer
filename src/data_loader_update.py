# data_loader.py
# Continuous (cross-year) per-watershed dataset for Transformer training
# X: (L, C, H, W) from HDF5 climate; y: vector of streamflow leads from CSV
# Static variables from a separate HDF5:
#   DEM:  (H_dem, W_dem)
#   awc:  (H_awc, W_awc)
#   fc:   (H_fc,  W_fc)
#   soil: (3, H_soil, W_soil) with channels [clay, sand, silt]
#
# Key features in this version:
# - GLOBAL min–max normalization (post-transform) fit on TRAIN basins only.
# - Transforms (e.g., log1p for prcp and Q) applied BEFORE scaling.
# - Validation/Test must load the same caches produced from the train split.
# - Backward-compat: If GLOBAL not present in cache, functions fall back to per-WS stats.

import os
import re
import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

# ============================================================
# Transforms
# ============================================================
# We'll use log10(x + 10e-3) = log10(x + 0.01)
_EPS_LOG10 = 10e-3  # 0.01

_VALID_TRANSFORMS = {None, "identity", "log10eps", "asinh", "sqrt"}

def _transform_array(arr, kind=None):
    """
    Apply a pointwise transform to a numpy array.
    Supported kinds: None/'identity', 'log10eps', 'asinh', 'sqrt'.

    - 'log10eps': good for zero-inflated, non-negative data (e.g., precip, flow).
                 Negatives are clipped to 0 prior to transform. Uses log10(x + 0.01).
    - 'asinh'   : good for signed data with heavy tails (keeps sign).
    - 'sqrt'    : good for non-negative data, milder than log transforms.
    """
    if kind in (None, "identity"):
        return arr.astype(np.float32)

    arr = arr.astype(np.float32)
    if kind == "log10eps":
        # clip negatives, add epsilon, then log10
        arr = np.where(arr < 0.0, 0.0, arr)
        return np.log10(arr + _EPS_LOG10)
    elif kind == "asinh":
        return np.arcsinh(arr)
    elif kind == "sqrt":
        arr = np.where(arr < 0.0, 0.0, arr)
        return np.sqrt(arr)
    else:
        raise ValueError(f"Unknown transform kind: {kind}")

def _inverse_transform_array(arr, kind=None):
    """Inverse of _transform_array for convenience (not used in dataset)."""
    if kind in (None, "identity"):
        return arr
    if kind == "log10eps":
        # inverse of log10(x + eps): 10**y - eps, clipped to >= 0
        out = (10.0 ** arr) - _EPS_LOG10
        return np.where(out < 0.0, 0.0, out)
    if kind == "asinh":
        return np.sinh(arr)
    if kind == "sqrt":
        return np.square(arr)
    raise ValueError(f"Unknown transform kind: {kind}")

# ============================================================
# Streamflow (CSV) loaders
# ============================================================

def load_streamflow_csv_by_year(csv_path, start_year=2000):
    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c) for c in df.columns]
    for dc in ('date', 'Date'):
        if dc in df.columns:
            df = df.drop(columns=[dc])

    n_rows = len(df)
    assert n_rows % 365 == 0, f"CSV rows ({n_rows}) not multiple of 365."
    n_years = n_rows // 365
    csv_years = [start_year + i for i in range(n_years)]

    flow_by_year = {}
    for col in df.columns:
        series = df[col].astype(float).to_numpy()
        arr = series.reshape(n_years, 365)
        flow_by_year[col] = {csv_years[i]: arr[i] for i in range(n_years)}
    return flow_by_year, csv_years

# ============================================================
# HDF5 scanning (ordered) for climate
# ============================================================

def _years_from_group(g):
    yr_re = re.compile(r"^(\d{4})subset\d+$")
    out = []
    for k in g.keys():
        m = yr_re.match(k)
        if m:
            out.append((int(m.group(1)), k))
    out.sort(key=lambda x: x[0])
    return out

def _scan_h5_ordered(h5_path, variables=None, watersheds=None, start_year=None, end_year=None):
    out = {}
    with h5py.File(h5_path, 'r') as f:
        groups = [k for k in f.keys() if k.endswith('_watershed')]
        if watersheds:
            ws_set = set(map(str, watersheds))
            groups = [g for g in groups if g.split('_')[0] in ws_set or g in ws_set]

        for ws in groups:
            var_map = {}
            years_sets = []

            for var in f[ws].keys():
                if not isinstance(f[ws][var], h5py.Group):
                    continue
                if variables and var not in variables:
                    continue

                yrs = _years_from_group(f[ws][var])
                if not yrs:
                    continue

                if start_year is not None or end_year is not None:
                    yrs = [(y, k) for (y, k) in yrs
                           if (start_year is None or y >= start_year)
                           and (end_year   is None or y <= end_year)]
                    if not yrs:
                        continue

                var_paths = [(y, f"{ws}/{var}/{ky}") for (y, ky) in yrs]
                var_map[var] = var_paths
                years_sets.append({y for y, _ in var_paths})

            if not var_map:
                continue

            common_years = None
            for s in years_sets:
                common_years = s if common_years is None else common_years.intersection(s)
            if not common_years:
                continue
            years = sorted(common_years)

            vars_paths_ordered = {}
            H = W = None
            for var, pairs in var_map.items():
                d = {y: p for y, p in pairs if y in common_years}
                ordered = [d[y] for y in years]
                if H is None:
                    T, H, W = f[ordered[0]].shape
                    assert T == 365, f"{ws}/{var} year has T={T}, expected 365"
                else:
                    T, h2, w2 = f[ordered[0]].shape
                    assert (T, h2, w2) == (365, H, W), f"Shape mismatch in {ws}/{var}"
                vars_paths_ordered[var] = ordered

            out[ws] = {
                'years': years,
                'vars': vars_paths_ordered,
                'shape': (H, W)
            }
    return out

# ============================================================
# Lead handling & index building
# ============================================================

def default_ws_name_to_flow_col(ws_name: str) -> str:
    return ws_name.split('_')[0]

def _normalize_leads(lead_days=None, horizon=1):
    if lead_days is None:
        return [int(horizon)]
    if isinstance(lead_days, int):
        assert lead_days >= 1, "lead_days int must be >= 1"
        return list(range(1, lead_days + 1))
    leads = sorted({int(x) for x in lead_days})
    assert len(leads) > 0 and leads[0] >= 1, "lead_days must contain positive integers"
    return leads

def _build_per_ws_meta(struct, flow_by_year, start_year=None, end_year=None):
    metas = {}
    for ws, meta in struct.items():
        flow_col = default_ws_name_to_flow_col(ws)
        if flow_col not in flow_by_year:
            continue

        years_struct = set(meta['years'])
        years_csv    = set(flow_by_year[flow_col].keys())

        if start_year is not None:
            years_struct = {y for y in years_struct if y >= start_year}
            years_csv    = {y for y in years_csv    if y >= start_year}
        if end_year is not None:
            years_struct = {y for y in years_struct if y <= end_year}
            years_csv    = {y for y in years_csv    if y <= end_year}

        years_common = sorted(years_struct.intersection(years_csv))
        if not years_common:
            continue

        var_year_paths = {}
        for var, ordered_paths in meta['vars'].items():
            pby = {y: p for y, p in zip(meta['years'], ordered_paths)}
            var_year_paths[var] = [pby[y] for y in years_common if y in pby]

        flow_vec = np.concatenate([flow_by_year[flow_col][y] for y in years_common], axis=0)

        metas[ws] = {
            'watershed': ws,
            'years_common': years_common,
            'shape': meta['shape'],
            'variables': list(var_year_paths.keys()),
            'var_year_paths': var_year_paths,
            'flow_col': flow_col,
            'flow_vec': flow_vec
        }
    return metas

def _num_windows_contiguous(total_days, seq_len, max_lead, stride):
    max_start = total_days - seq_len - max_lead
    if max_start < 0:
        return 0
    return (max_start // stride) + 1

def _build_index_contiguous_from_meta(per_ws_meta, seq_len, stride, lead_days,
                                      variables=None, drop_nan_targets=True):
    lead_list = _normalize_leads(lead_days, horizon=1)
    max_lead = max(lead_list) - 1

    index = []
    for ws, meta in per_ws_meta.items():
        vars_here = variables if variables else meta['variables']
        vars_here = [v for v in vars_here if v in meta['var_year_paths']]
        if not vars_here:
            continue

        H, W = meta['shape']
        flow_vec = meta['flow_vec']
        total_days = flow_vec.shape[0]

        n_win = _num_windows_contiguous(total_days, seq_len, max_lead, stride)
        if n_win == 0:
            continue

        for k in range(n_win):
            s = k * stride
            if drop_nan_targets:
                t0 = s + seq_len - 1
                targets = [flow_vec[t0 + (d - 1)] for d in lead_list]
                if any((np.isnan(v) or np.isinf(v)) for v in targets):
                    continue

            index.append({
                'watershed': ws,
                'flow_col': meta['flow_col'],
                'start_global': s,
                'seq_len': seq_len,
                'H': H, 'W': W,
                'variables': vars_here,
                'var_year_paths': {v: meta['var_year_paths'][v] for v in vars_here},
                'lead_days': lead_list
            })
    return index

# ============================================================
# HDF slicing across years (climate)
# ============================================================

def _slice_across_years(h5f, paths_by_year, start_global, length):
    T, H, W = h5f[paths_by_year[0]].shape
    assert T == 365, "Each per-year dataset must have 365 days"
    out = np.empty((length, H, W), dtype=np.float32)

    remaining = length
    pos = 0
    t = start_global
    while remaining > 0:
        yr = t // 365
        off = t % 365
        ds = h5f[paths_by_year[yr]]
        take = min(365 - off, remaining)
        out[pos:pos+take] = ds[off:off+take]
        t += take
        pos += take
        remaining -= take
    return out

# ============================================================
# Min/Max calculators (GLOBAL and legacy)
# ============================================================

def calculate_global_min_max_climate_global(
    h5_file_path,
    transform_map=None,
    watersheds=None,
    variables=None,
    start_year=None,
    end_year=None,
):
    """Return {'GLOBAL': {var: {'min': m, 'max': M}}} across selected watersheds/years AFTER transform."""
    transform_map = transform_map or {}
    mins = {}
    maxs = {}
    with h5py.File(h5_file_path, 'r') as f:
        all_ws = [k for k in f.keys() if k.endswith('_watershed')]
        if watersheds:
            ws_set = set(map(str, watersheds))
            all_ws = [g for g in all_ws if g.split('_')[0] in ws_set or g in ws_set]

        yr_re = re.compile(r"^(\d{4})subset\d+$")
        for ws in all_ws:
            for var in f[ws].keys():
                if not isinstance(f[ws][var], h5py.Group):
                    continue
                if variables and var not in variables:
                    continue

                pairs = []
                for ky in f[ws][var].keys():
                    m = yr_re.match(ky)
                    if not m:
                        continue
                    y = int(m.group(1))
                    if (start_year is not None and y < start_year) or (end_year is not None and y > end_year):
                        continue
                    pairs.append((y, ky))
                pairs.sort(key=lambda x: x[0])
                if not pairs:
                    continue

                tr = transform_map.get(var, None)
                for _, ky in pairs:
                    arr = f[ws][var][ky][()]
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    arr = _transform_array(arr, tr)
                    cur_min = float(np.min(arr))
                    cur_max = float(np.max(arr))
                    mins[var] = cur_min if var not in mins else min(mins[var], cur_min)
                    maxs[var] = cur_max if var not in maxs else max(maxs[var], cur_max)

    out = {'GLOBAL': {}}
    for v in mins.keys():
        out['GLOBAL'][v] = {'min': float(mins[v]), 'max': float(maxs[v])}
    return out

def calculate_global_min_max_climate(h5_file_path, transform_map=None):
    """
    Legacy per-watershed min/max AFTER transform (kept for backward-compat).
    Returns {ws: {var: {'min','max'}}}
    """
    transform_map = transform_map or {}
    with h5py.File(h5_file_path, 'r') as f:
        global_min_max = {}
        for watershed in f:
            watershed_data = {}
            for variable in f[watershed]:
                if not isinstance(f[watershed][variable], h5py.Group):
                    continue
                variable_data = []
                for year in f[watershed][variable]:
                    year_data = f[watershed][variable][year][()]
                    year_data = np.nan_to_num(year_data, nan=0.0, posinf=0.0, neginf=0.0)
                    tr = transform_map.get(variable, None)
                    year_data = _transform_array(year_data, tr)
                    variable_data.append(year_data)
                if not variable_data:
                    continue
                variable_data = np.concatenate(variable_data, axis=0)
                min_val = float(np.min(variable_data))
                max_val = float(np.max(variable_data))
                watershed_data[variable] = {'min': min_val, 'max': max_val}
            if watershed_data:
                global_min_max[watershed] = watershed_data
    return global_min_max

def calculate_global_min_max_streamflow_global(
    csv_path,
    watersheds=None,      # list like ['04127800', ...]
    start_year=2000,
    transform_kind="log1p"
):
    """Return {'GLOBAL': {'min': m, 'max': M}} across selected WS flows AFTER transform."""
    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c) for c in df.columns]
    for dc in ('date', 'Date'):
        if dc in df.columns:
            df = df.drop(columns=[dc])

    if watersheds:
        keep = set(map(str, watersheds))
        cols = [c for c in df.columns if c in keep]
        if cols:
            df = df[cols]

    series = df.astype(float).to_numpy()
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    series = _transform_array(series, transform_kind)

    return {'GLOBAL': {'min': float(np.min(series)), 'max': float(np.max(series))}}

def calculate_global_min_max_streamflow(csv_path, start_year=2000, transform_kind=None):
    """
    Legacy per-watershed streamflow min/max AFTER transform.
    Returns {ws+'_watershed': {'min','max'}}
    """
    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c) for c in df.columns]
    for dc in ('date', 'Date'):
        if dc in df.columns:
            df = df.drop(columns=[dc])

    global_min_max = {}
    n_rows = len(df)
    assert n_rows % 365 == 0, f"CSV rows ({n_rows}) not multiple of 365."
    n_years = n_rows // 365

    for ws in df.columns:
        series = df[ws].astype(float).to_numpy().reshape(n_years, 365)
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
        series = _transform_array(series, transform_kind)
        min_val = float(np.min(series))
        max_val = float(np.max(series))
        global_min_max[ws + "_watershed"] = {'min': min_val, 'max': max_val}
    return global_min_max

# ---- Static min/max helpers --------------------------------

def _safe_minmax(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return float(np.min(arr)), float(np.max(arr))

def _pick_first_dataset(grp, prefer_suffix=None):
    """Return first dataset in a group; if prefer_suffix is given, prefer names ending with it."""
    if prefer_suffix is not None:
        for k in grp.keys():
            obj = grp[k]
            if isinstance(obj, h5py.Dataset) and k.lower().endswith(prefer_suffix.lower()):
                return obj
    for k in grp.keys():
        obj = grp[k]
        if isinstance(obj, h5py.Dataset):
            return obj
    return None

def calculate_global_min_max_static(static_h5_path):
    """
    Legacy per-watershed static min/max.
    Returns {ws: {'DEM':{'min','max'}, 'awc':..., 'fc':..., 'soil':{'clay':...}}}
    """
    out = {}
    with h5py.File(static_h5_path, 'r') as f:
        for ws in f.keys():
            g = f[ws]
            ws_out = {}

            # DEM
            if 'DEM_clips' in g and isinstance(g['DEM_clips'], h5py.Group):
                dem_ds = _pick_first_dataset(g['DEM_clips'])
                if dem_ds is not None:
                    arr = dem_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    mn, mx = _safe_minmax(arr)
                    ws_out['DEM'] = {'min': mn, 'max': mx}

            # awc
            if 'awc_clips' in g and isinstance(g['awc_clips'], h5py.Group):
                awc_ds = _pick_first_dataset(g['awc_clips'], prefer_suffix="_awc")
                if awc_ds is not None:
                    arr = awc_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    mn, mx = _safe_minmax(arr)
                    ws_out['awc'] = {'min': mn, 'max': mx}

            # fc
            if 'fc_clips' in g and isinstance(g['fc_clips'], h5py.Group):
                fc_ds = _pick_first_dataset(g['fc_clips'], prefer_suffix="_fc")
                if fc_ds is not None:
                    arr = fc_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    mn, mx = _safe_minmax(arr)
                    ws_out['fc'] = {'min': mn, 'max': mx}

            # SOIL
            if 'soil_clips' in g and isinstance(g['soil_clips'], h5py.Group):
                soil_grp = g['soil_clips']
                soil_out = {}
                for k in soil_grp.keys():
                    obj = soil_grp[k]
                    if not isinstance(obj, h5py.Dataset):
                        continue
                    sub = k.split('_')[-1].lower()  # clay/sand/silt
                    if sub not in ('clay', 'sand', 'silt'):
                        continue
                    arr = obj[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    mn, mx = _safe_minmax(arr)
                    soil_out[sub] = {'min': mn, 'max': mx}
                if soil_out:
                    ws_out['soil'] = soil_out

            if ws_out:
                out[ws] = ws_out
    return out

def calculate_global_min_max_static_global(static_h5_path, watersheds=None):
    """
    Return {'GLOBAL': {'DEM':{'min','max'}, 'awc':..., 'fc':..., 'soil':{'clay':...,'sand':...,'silt':...}}}
    across selected watersheds.
    """
    mins = {'DEM': None, 'awc': None, 'fc': None, 'soil': {'clay': None, 'sand': None, 'silt': None}}
    maxs = {'DEM': None, 'awc': None, 'fc': None, 'soil': {'clay': None, 'sand': None, 'silt': None}}

    def _accum(key, val_min, val_max, sub=None):
        if sub is None:
            mins[key] = val_min if mins[key] is None else min(mins[key], val_min)
            maxs[key] = val_max if maxs[key] is None else max(maxs[key], val_max)
        else:
            mins[key][sub] = val_min if mins[key][sub] is None else min(mins[key][sub], val_min)
            maxs[key][sub] = val_max if maxs[key][sub] is None else max(maxs[key][sub], val_max)

    with h5py.File(static_h5_path, 'r') as f:
        all_ws = list(f.keys())
        if watersheds:
            ws_set = set(map(str, watersheds))
            all_ws = [ws for ws in all_ws if ws in ws_set or ws.split('_')[0] in ws_set]

        for ws in all_ws:
            g = f[ws]

            # DEM
            if 'DEM_clips' in g and isinstance(g['DEM_clips'], h5py.Group):
                dem_ds = _pick_first_dataset(g['DEM_clips'])
                if dem_ds is not None:
                    arr = dem_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    mn, mx = _safe_minmax(arr)
                    _accum('DEM', mn, mx)

            # awc
            if 'awc_clips' in g and isinstance(g['awc_clips'], h5py.Group):
                awc_ds = _pick_first_dataset(g['awc_clips'], prefer_suffix="_awc")
                if awc_ds is not None:
                    arr = awc_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    mn, mx = _safe_minmax(arr)
                    _accum('awc', mn, mx)

            # fc
            if 'fc_clips' in g and isinstance(g['fc_clips'], h5py.Group):
                fc_ds = _pick_first_dataset(g['fc_clips'], prefer_suffix="_fc")
                if fc_ds is not None:
                    arr = fc_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    mn, mx = _safe_minmax(arr)
                    _accum('fc', mn, mx)

            # soil
            if 'soil_clips' in g and isinstance(g['soil_clips'], h5py.Group):
                soil_grp = g['soil_clips']
                for k in soil_grp.keys():
                    obj = soil_grp[k]
                    if not isinstance(obj, h5py.Dataset):
                        continue
                    sub = k.split('_')[-1].lower()
                    if sub not in ('clay', 'sand', 'silt'):
                        continue
                    arr = obj[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    mn, mx = _safe_minmax(arr)
                    _accum('soil', mn, mx, sub=sub)

    out = {'GLOBAL': {
        'DEM': {'min': float(mins['DEM']) if mins['DEM'] is not None else 0.0,
                'max': float(maxs['DEM']) if maxs['DEM'] is not None else 1.0},
        'awc': {'min': float(mins['awc']) if mins['awc'] is not None else 0.0,
                'max': float(maxs['awc']) if maxs['awc'] is not None else 1.0},
        'fc':  {'min': float(mins['fc'])  if mins['fc']  is not None else 0.0,
                'max': float(maxs['fc'])  if maxs['fc']  is not None else 1.0},
        'soil': {
            'clay': {'min': float(mins['soil']['clay']) if mins['soil']['clay'] is not None else 0.0,
                     'max': float(maxs['soil']['clay']) if maxs['soil']['clay'] is not None else 1.0},
            'sand': {'min': float(mins['soil']['sand']) if mins['soil']['sand'] is not None else 0.0,
                     'max': float(maxs['soil']['sand']) if maxs['soil']['sand'] is not None else 1.0},
            'silt': {'min': float(mins['soil']['silt']) if mins['soil']['silt'] is not None else 0.0,
                     'max': float(maxs['soil']['silt']) if maxs['soil']['silt'] is not None else 1.0},
        }
    }}
    return out

# ============================================================
# Normalizers (prefer GLOBAL if present)
# ============================================================

def _norm_array(arr, min_val, max_val):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1.0
    return (arr - min_val) / denom

def _mm_key_for_ws(gmm_dict, ws_name):
    """Return the key present in gmm_dict matching ws_name or its ID without suffix."""
    if ws_name in gmm_dict:
        return ws_name
    ws_id = ws_name.split('_')[0]
    if ws_id in gmm_dict:
        return ws_id
    return ws_name  # may raise later if truly missing

def normalize_streamflow_data(data, global_min_max_streamflow, watershed=None):
    mm = global_min_max_streamflow.get('GLOBAL', None)
    if mm is None:
        mm = global_min_max_streamflow[watershed]
    denom = (mm['max'] - mm['min']) if (mm['max'] - mm['min']) != 0 else 1.0
    return (data - mm['min']) / denom

def normalize_climate_data(data, global_min_max, watershed, variable):
    if 'GLOBAL' in global_min_max and variable in global_min_max['GLOBAL']:
        mm = global_min_max['GLOBAL'][variable]
    else:
        mm = global_min_max[watershed][variable]
    denom = (mm['max'] - mm['min']) if (mm['max'] - mm['min']) != 0 else 1.0
    return (data - mm['min']) / denom

def normalize_static_entry(arr, gmm_static, mm_ws_key, key, subkey=None):
    if 'GLOBAL' in gmm_static:
        if key == 'soil':
            mm = gmm_static['GLOBAL']['soil'][subkey]
        else:
            mm = gmm_static['GLOBAL'][key]
    else:
        if key == 'soil':
            mm = gmm_static[mm_ws_key]['soil'][subkey]
        else:
            mm = gmm_static[mm_ws_key][key]
    return _norm_array(arr, mm['min'], mm['max'])

# ============================================================
# JSON helpers
# ============================================================

def save_min_max_to_json(global_min_max, filename):
    def convert_np_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    def recurse(x):
        if isinstance(x, dict):
            return {k: recurse(v) for k, v in x.items()}
        return convert_np_types(x)

    with open(filename, 'w') as f:
        json.dump(recurse(global_min_max), f, indent=4)

def load_min_max_from_json(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# ============================================================
# Dataset
# ============================================================

class WatershedFlowDataset(Dataset):
    def __init__(self, h5_path, csv_path, static_h5,
                 variables=None, watersheds=None,
                 seq_len=120, stride=1,
                 lead_days=None, horizon=1,
                 start_year=2000, end_year=None,
                 drop_nan_targets=True,
                 # Transforms
                 climate_transform_map=None,   # dict: {var_name: 'log1p'|'asinh'|'sqrt'|None}
                 streamflow_transform="log1p", # default to log1p for flows
                 # Min/max cache files (use split-specific filenames)
                 min_max_file_climate="mm__CLIMATE__TRAIN_GLOBAL.json",
                 min_max_file_streamflow="mm__FLOW__TRAIN_GLOBAL.json",
                 min_max_file_static="mm__STATIC__TRAIN_GLOBAL.json",
                 # GLOBAL fitting scope
                 min_max_scope="global",       # "global" (recommended) or "legacy"
                 mm_watersheds=None):          # list of WS IDs used to FIT min–max (pass train WS here)
        """
        Set climate_transform_map only for variables that need it (e.g., {'prcp':'log1p'}).
        Streamflow transform defaults to 'log1p'. Use None/'identity' to disable.
        For TRUE ungauged evaluation:
          - Fit caches on TRAIN only (pass mm_watersheds=train_ws).
          - Use the SAME cache files for val/test (mm_watersheds=None so they just load).
        """
        # Validate transforms
        self.climate_transform_map = {}
        if climate_transform_map:
            for k, v in climate_transform_map.items():
                if v not in _VALID_TRANSFORMS:
                    raise ValueError(f"Bad transform '{v}' for climate var '{k}'")
                self.climate_transform_map[k] = v
        self.streamflow_transform = streamflow_transform
        if streamflow_transform not in _VALID_TRANSFORMS:
            raise ValueError(f"Bad streamflow transform '{streamflow_transform}'")

        self.h5_path = h5_path
        self._h5 = None
        self.static_h5 = h5py.File(static_h5, 'r')

        self.min_max_file_climate = min_max_file_climate
        self.min_max_file_streamflow = min_max_file_streamflow
        self.min_max_file_static = min_max_file_static

        # CSV by-year dict
        flow_by_year, _ = load_streamflow_csv_by_year(csv_path, start_year=start_year)

        # Climate HDF structure (ordered paths)
        struct = _scan_h5_ordered(h5_path, variables, watersheds,
                                  start_year=start_year, end_year=end_year)

        # Per-watershed meta (years intersection + concatenated flow)
        per_ws_meta = _build_per_ws_meta(struct, flow_by_year,
                                         start_year=start_year, end_year=end_year)

        # ---- Min/Max caches (fit on TRAIN basins if files not present)
        fit_ws = mm_watersheds if mm_watersheds is not None else watersheds

        # Climate min/max (GLOBAL, transformed domain)
        self.global_min_max_climate = load_min_max_from_json(self.min_max_file_climate)
        if self.global_min_max_climate is None:
            if min_max_scope == "global":
                self.global_min_max_climate = calculate_global_min_max_climate_global(
                    h5_file_path=h5_path,
                    transform_map=self.climate_transform_map,
                    watersheds=fit_ws,
                    variables=variables,
                    start_year=start_year,
                    end_year=end_year,
                )
            else:
                self.global_min_max_climate = calculate_global_min_max_climate(
                    h5_path, transform_map=self.climate_transform_map
                )
            save_min_max_to_json(self.global_min_max_climate, self.min_max_file_climate)
        else:
            print("[INFO] Loaded climate min/max from cache. Ensure it matches current transform map and split.")

        # Streamflow min/max (GLOBAL)
        self.global_min_max_streamflow = load_min_max_from_json(self.min_max_file_streamflow)
        if self.global_min_max_streamflow is None:
            if min_max_scope == "global":
                fit_cols = None
                if fit_ws:
                    fit_cols = [str(w).split('_')[0] for w in fit_ws]
                self.global_min_max_streamflow = calculate_global_min_max_streamflow_global(
                    csv_path,
                    watersheds=fit_cols,
                    start_year=start_year,
                    transform_kind=self.streamflow_transform
                )
            else:
                self.global_min_max_streamflow = calculate_global_min_max_streamflow(
                    csv_path, start_year=start_year, transform_kind=self.streamflow_transform
                )
            save_min_max_to_json(self.global_min_max_streamflow, self.min_max_file_streamflow)
        else:
            print("[INFO] Loaded streamflow min/max from cache. Ensure it matches current transform.")

        # Static min/max (GLOBAL)
        self.global_min_max_static = load_min_max_from_json(self.min_max_file_static)
        if self.global_min_max_static is None:
            if min_max_scope == "global":
                self.global_min_max_static = calculate_global_min_max_static_global(static_h5, watersheds=fit_ws)
            else:
                self.global_min_max_static = calculate_global_min_max_static(static_h5)
            save_min_max_to_json(self.global_min_max_static, self.min_max_file_static)

        # Index for sampling windows
        self.lead_list = _normalize_leads(lead_days, horizon=horizon)
        self.index = _build_index_contiguous_from_meta(
            per_ws_meta, seq_len, stride, self.lead_list,
            variables=variables, drop_nan_targets=drop_nan_targets
        )

        self.per_ws_meta = per_ws_meta
        self.ws_of = [it["watershed"] for it in self.index]

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r', libver='latest', swmr=False)

    def __len__(self):
        return len(self.index)

    def _static_group_for_ws(self, ws_name):
        """Prefer exact match; fallback to no-suffix ws id if needed."""
        if ws_name in self.static_h5:
            return self.static_h5[ws_name]
        ws_id = ws_name.split('_')[0]
        return self.static_h5.get(ws_id, None)

    def __getitem__(self, i):
        self._ensure_open()
        it = self.index[i]
        L, H, W = it["seq_len"], it["H"], it["W"]
        C = len(it["variables"])
        s = it["start_global"]

        # ----- X (climate)
        X = np.empty((L, C, H, W), dtype=np.float32)
        for c, v in enumerate(it["variables"]):
            paths = it["var_year_paths"][v]
            Xi = _slice_across_years(self._h5, paths, s, L)
            Xi = np.nan_to_num(Xi, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply configured transform BEFORE normalizing
            tr_kind = self.climate_transform_map.get(v, None)
            Xi = _transform_array(Xi, tr_kind)

            # Normalize with GLOBAL stats (if available)
            Xi = normalize_climate_data(Xi, self.global_min_max_climate, it["watershed"], v)
            X[:, c] = Xi

        # ----- Static
        ws_name = it["watershed"]
        g = self._static_group_for_ws(ws_name)
        DEM = awc = fc = None
        soil = None

        if g is not None:
            mm_ws = 'GLOBAL' if 'GLOBAL' in self.global_min_max_static else _mm_key_for_ws(self.global_min_max_static, ws_name)

            # DEM
            if ('DEM' in self.global_min_max_static.get(mm_ws, {}) if mm_ws != 'GLOBAL' else True) and 'DEM_clips' in g and isinstance(g['DEM_clips'], h5py.Group):
                dem_ds = _pick_first_dataset(g['DEM_clips'])
                if dem_ds is not None:
                    arr = dem_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    DEM = torch.from_numpy(normalize_static_entry(arr, self.global_min_max_static, mm_ws, 'DEM'))

            # awc
            if ('awc' in self.global_min_max_static.get(mm_ws, {}) if mm_ws != 'GLOBAL' else True) and 'awc_clips' in g and isinstance(g['awc_clips'], h5py.Group):
                awc_ds = _pick_first_dataset(g['awc_clips'], prefer_suffix="_awc")
                if awc_ds is not None:
                    arr = awc_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    awc = torch.from_numpy(normalize_static_entry(arr, self.global_min_max_static, mm_ws, 'awc'))

            # fc
            if ('fc' in self.global_min_max_static.get(mm_ws, {}) if mm_ws != 'GLOBAL' else True) and 'fc_clips' in g and isinstance(g['fc_clips'], h5py.Group):
                fc_ds = _pick_first_dataset(g['fc_clips'], prefer_suffix="_fc")
                if fc_ds is not None:
                    arr = fc_ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    fc = torch.from_numpy(normalize_static_entry(arr, self.global_min_max_static, mm_ws, 'fc'))

            # soil: stack [clay, sand, silt] -> (3, H, W), crop to min H/W if needed
            if ('soil' in self.global_min_max_static.get(mm_ws, {}) if mm_ws != 'GLOBAL' else True) and 'soil_clips' in g and isinstance(g['soil_clips'], h5py.Group):
                soil_grp = g['soil_clips']
                chans = []
                order = ['clay', 'sand', 'silt']
                for sub in order:
                    ds = None
                    for k in soil_grp.keys():
                        obj = soil_grp[k]
                        if isinstance(obj, h5py.Dataset) and k.lower().endswith(f"_{sub}"):
                            ds = obj
                            break
                    if ds is None:
                        continue
                    arr = ds[()]
                    if arr.ndim == 3 and arr.shape[0] == 1: arr = arr[0]
                    arr = normalize_static_entry(arr, self.global_min_max_static, mm_ws, 'soil', subkey=sub)
                    chans.append(arr.astype(np.float32))

                if len(chans) > 0:
                    Hs = [c.shape[0] for c in chans]
                    Ws = [c.shape[1] for c in chans]
                    Hc, Wc = min(Hs), min(Ws)
                    chans = [c[:Hc, :Wc] for c in chans]
                    soil = torch.from_numpy(np.stack(chans, axis=0))  # (C<=3, Hc, Wc)

        # ----- y (streamflow targets)
        flow_vec = self.per_ws_meta[ws_name]['flow_vec']
        t0 = s + L - 1
        y_vec = np.array([flow_vec[t0 + (d - 1)] for d in it["lead_days"]], dtype=np.float32)
        y_vec = np.nan_to_num(y_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Transform BEFORE normalization (log1p by default)
        y_vec = _transform_array(y_vec, self.streamflow_transform)
        y_vec = normalize_streamflow_data(y_vec, self.global_min_max_streamflow, ws_name)

        return {
            "seq_id": i,
            "X": torch.from_numpy(X),
            "DEM": DEM,
            "awc": awc,
            "fc":  fc,
            "soil": soil,  # (C,H,W) or None
            "y": torch.from_numpy(y_vec),
            "meta": {
                "watershed": ws_name,
                "flow_col": it["flow_col"],
                "start_global": s,
                "seq_len": L,
                "lead_days": it["lead_days"],
                "variables": it["variables"],
                "H": H, "W": W
            }
        }

    def __del__(self):
        try:
            if self._h5:
                self._h5.close()
        except Exception:
            pass
        try:
            if self.static_h5:
                self.static_h5.close()
        except Exception:
            pass

# ============================================================
# Sampler & Collate
# ============================================================

class GroupedBatchSampler(Sampler):
    """Each batch contains samples from a single watershed (same H,W)."""
    def __init__(self, dataset: WatershedFlowDataset, batch_size=8, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        buckets = {}
        for i, ws in enumerate(dataset.ws_of):
            buckets.setdefault(ws, []).append(i)
        self.buckets = list(buckets.values())

    def __iter__(self):
        rng = np.random.default_rng()
        if self.shuffle:
            for b in self.buckets:
                rng.shuffle(b)
            rng.shuffle(self.buckets)
        for b in self.buckets:
            for k in range(0, len(b), self.batch_size):
                yield b[k:k + self.batch_size]

    def __len__(self):
        total = 0
        for b in self.buckets:
            total += (len(b) + self.batch_size - 1) // self.batch_size
        return total

def per_ws_collate(batch):
    """
    X -> (B,L,C,H,W), y -> (B,K), metas list
    Static variables are returned ONCE per batch (taken from the first item),
    because GroupedBatchSampler guarantees all items in a batch share the same watershed.
    """
    X = torch.stack([b["X"] for b in batch], 0)
    y = torch.stack([b["y"] for b in batch], 0)
    metas = [b["meta"] for b in batch]

    # Static tensors (no lists)
    DEM  = batch[0].get("DEM", None)
    awc  = batch[0].get("awc", None)
    fc   = batch[0].get("fc", None)
    soil = batch[0].get("soil", None)

    return {"X": X, "DEM": DEM, "awc": awc, "fc": fc, "soil": soil, "y": y, "meta": metas}

# ============================================================
# Validators / Utilities
# ============================================================

def _shape_or_none(t):
    try:
        return tuple(t.shape)
    except Exception:
        return None

def validate_items(dataset, n=5):
    print(f"Dataset size: {len(dataset)}")
    step = max(1, len(dataset)//n) if len(dataset) > n else 1
    checked = 0
    for i in range(0, len(dataset), step):
        sample = dataset[i]
        X, y, meta, seq_id = sample["X"], sample["y"], sample["meta"], sample["seq_id"]
        DEM, awc, fc, soil = sample.get("DEM"), sample.get("awc"), sample.get("fc"), sample.get("soil")
        L, C, H, W = X.shape
        leads = meta['lead_days']
        y_np = y.numpy()
        preview = np.array2string(y_np[:min(5, len(y_np))], precision=3, separator=', ')
        print(f"[ITEM] seq_id={seq_id} | ws={meta['watershed']} | start_global={meta['start_global']} "
              f"| L={L} | leads={leads}")
        print(f"       X.shape={tuple(X.shape)} (L,C,H,W) | y.shape={tuple(y.shape)} values={preview}")
        print(f"       DEM:{_shape_or_none(DEM)} | awc:{_shape_or_none(awc)} | fc:{_shape_or_none(fc)} "
              f"| soil:{_shape_or_none(soil)}")
        print(f"       variables={meta['variables']}")
        checked += 1
        if checked >= n:
            break

def validate_batches(loader, n_batches=2):
    from collections import Counter
    seen = 0
    for batch in loader:
        X, y, metas = batch["X"], batch["y"], batch["meta"]
        B, L, C, H, W = X.shape
        ws_ids = [m["watershed"] for m in metas]
        cnt = Counter(ws_ids)

        DEM  = batch.get("DEM")
        awc  = batch.get("awc")
        fc   = batch.get("fc")
        soil = batch.get("soil")

        print(f"[BATCH] size={B} | X.shape={tuple(X.shape)} | y.shape={tuple(y.shape)} "
              f"| watersheds_in_batch={dict(cnt)}")
        print(f"        DEM:{_shape_or_none(DEM)} | awc:{_shape_or_none(awc)} | fc:{_shape_or_none(fc)} | soil:{_shape_or_none(soil)}")
        assert len(cnt) == 1, "Batch spans multiple watersheds (should not)."
        seen += 1
        if seen >= n_batches:
            break

# ============================================================
# Example usage (run this file directly)
# ============================================================

if __name__ == "__main__":
    # ---- paths (edit to your machine)
    H5_PATH   = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
    CSV_PATH  = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"
    STATIC_H5 = "/home/talhamuh/water-research/HydroTransformer/data/processed/static_parameters_data/file.h5"

    # ---- split lists (example)
    train_ws = None  # e.g., ['04127800','04119300', ...]  # Fit scalers on these
    val_ws   = None  # e.g., ['04142000', ...]

    # ---- config
    variables  = None            # or list like ['prcp','tmin','tmax','srad', ...]
    watersheds = train_ws
    seq_len    = 30
    stride     = 1
    lead_days  = 1

    start_year = 2000
    end_year   = 2021

    climate_transform_map = {
        # Example: adapt keys to your HDF variable names
        'prcp': 'log10eps', 'ppt': 'log10eps', 'precip': 'log10eps',
        # 'tmin': 'identity',
        # 'tmax': 'identity',
    }
    streamflow_transform = "log10eps"

    # Cache files (use the SAME three files for train/val/test; generate them from TRAIN first)
    mm_clim = "log10__CLIMATE__TRAIN_GLOBAL.json"
    mm_flow = "log10__FLOW__TRAIN_GLOBAL.json"
    mm_stat = "log10__STATIC__TRAIN_GLOBAL.json"

    dataset = WatershedFlowDataset(
        h5_path=H5_PATH,
        csv_path=CSV_PATH,
        static_h5=STATIC_H5,
        variables=variables,
        watersheds=watersheds,
        seq_len=seq_len, stride=stride,
        lead_days=lead_days,
        start_year=start_year, end_year=end_year,
        drop_nan_targets=True,
        climate_transform_map=climate_transform_map,
        streamflow_transform=streamflow_transform,
        min_max_file_climate=mm_clim,
        min_max_file_streamflow=mm_flow,
        min_max_file_static=mm_stat,
        min_max_scope="global",
        mm_watersheds=train_ws,  # <-- FIT GLOBAL stats on train basins only
    )

    sampler = GroupedBatchSampler(dataset, batch_size=4, shuffle=True)
    loader  = DataLoader(dataset, batch_sampler=sampler,
                         num_workers=4, pin_memory=True,
                         collate_fn=per_ws_collate)
    print(f"Total dataset size: {len(dataset)} sequences in {len(sampler)} batches.")
    validate_batches(loader, n_batches=2)

    batch = next(iter(loader))
    X, y = batch["X"].float(), batch["y"].float()
    DEM, awc, fc, soil = batch["DEM"], batch["awc"], batch["fc"], batch["soil"]
    print("Example batch tensors:", X.shape, y.shape)
    def _shape_or_none(t):
        try: return tuple(t.shape)
        except: return None
    print("Static shapes:", _shape_or_none(DEM), _shape_or_none(awc), _shape_or_none(fc), _shape_or_none(soil))
