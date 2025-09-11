# data_loading.py
# Continuous (cross-year) per-watershed dataset for Transformer training
# X: (L, C, H, W) from HDF5 climate; y: vector of streamflow leads from CSV

import os
import re
import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


# =========================
# Streamflow (CSV) loaders
# =========================

def load_streamflow_csv_by_year(csv_path, start_year=2000):
    """
    CSV format:
      - Header row has watershed IDs as columns (optionally a 'date'/'Date' column which is dropped).
      - Rows are daily values from 1/1/start_year onward.
      - Exactly 365 rows per year (leap days trimmed).

    Returns:
      flow_by_year: dict[str ws_id] -> dict[int year] -> np.ndarray (365,)
      csv_years: sorted list of years present in CSV (inferred from rows)
    """
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
        arr = series.reshape(n_years, 365)  # (n_years, 365)
        flow_by_year[col] = {csv_years[i]: arr[i] for i in range(n_years)}
    return flow_by_year, csv_years


# =========================
# HDF5 scanning (ordered)
# =========================

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
    """
    Build an ordered view of the HDF5:
      returns dict:
        ws -> {
          'years': [y0, y1, ...]  # ordered and filtered to [start_year, end_year] if provided
          'vars': { var -> [dataset_path per each year in 'years'] },
          'shape': (H, W)
        }

    Only includes variables/years that are present for *all* selected variables within a watershed.
    """
    out = {}
    with h5py.File(h5_path, 'r') as f:
        groups = [k for k in f.keys() if k.endswith('_watershed')]
        if watersheds:
            ws_set = set(map(str, watersheds))
            groups = [g for g in groups if g.split('_')[0] in ws_set]

        for ws in groups:
            var_map = {}
            years_sets = []

            for var in f[ws].keys():
                if not isinstance(f[ws][var], h5py.Group):
                    continue
                if variables and var not in variables:
                    continue

                yrs = _years_from_group(f[ws][var])  # [(year_int, key)]
                if not yrs:
                    continue

                # year filter
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

            # intersect years across variables
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


# =========================
# Lead handling & index building
# =========================

def default_ws_name_to_flow_col(ws_name: str) -> str:
    """'4096405_watershed' -> '4096405'"""
    return ws_name.split('_')[0]

def _normalize_leads(lead_days=None, horizon=1):
    """
    Returns a sorted unique list of positive integer leads.
    - If lead_days is an int k: returns [1, 2, ..., k] (contiguous multi-step)
    - If lead_days is an iterable: sorted unique list(list(lead_days))
    - If lead_days is None: returns [horizon] (backward compatible)
    """
    if lead_days is None:
        return [int(horizon)]
    if isinstance(lead_days, int):
        assert lead_days >= 1, "lead_days int must be >= 1"
        return list(range(1, lead_days + 1))
    leads = sorted({int(x) for x in lead_days})
    assert len(leads) > 0 and leads[0] >= 1, "lead_days must contain positive integers"
    return leads

def _build_per_ws_meta(struct, flow_by_year, start_year=None, end_year=None):
    """
    For each watershed present in both HDF struct and CSV, produce:
      meta dict with:
        'watershed', 'years_common', 'shape', 'variables',
        'var_year_paths' (aligned to years_common),
        'flow_col', 'flow_vec' (concatenated per-year streamflow)
    """
    metas = {}
    for ws, meta in struct.items():
        flow_col = default_ws_name_to_flow_col(ws)
        if flow_col not in flow_by_year:
            continue

        years_struct = set(meta['years'])
        years_csv    = set(flow_by_year[flow_col].keys())

        # apply bounds again (no-op if already applied upstream)
        if start_year is not None:
            years_struct = {y for y in years_struct if y >= start_year}
            years_csv    = {y for y in years_csv    if y >= start_year}
        if end_year is not None:
            years_struct = {y for y in years_struct if y <= end_year}
            years_csv    = {y for y in years_csv    if y <= end_year}

        years_common = sorted(years_struct.intersection(years_csv))
        if not years_common:
            continue

        # Reorder per-variable paths to years_common
        var_year_paths = {}
        for var, ordered_paths in meta['vars'].items():
            pby = {y: p for y, p in zip(meta['years'], ordered_paths)}
            var_year_paths[var] = [pby[y] for y in years_common if y in pby]

        # Concatenate streamflow per-year chunks
        flow_vec = np.concatenate([flow_by_year[flow_col][y] for y in years_common], axis=0)  # (365 * len,)

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
    """
    Build a flat index of windows (contiguous across years) from per-watershed meta.
    lead_days: list[int] of leads (e.g., [1,3,7]) or contiguous [1..K]
    Drops windows if *any* of the target leads is NaN/inf (configurable here).
    """
    lead_list = _normalize_leads(lead_days, horizon=1)
    max_lead = max(lead_list)

    index = []
    for ws, meta in per_ws_meta.items():
        vars_here = variables if variables else meta['variables']
        vars_here = [v for v in vars_here if v in meta['var_year_paths']]
        if not vars_here:
            continue

        H, W = meta['shape']
        flow_vec = meta['flow_vec']  # shape: total_days
        total_days = flow_vec.shape[0]

        n_win = _num_windows_contiguous(total_days, seq_len, max_lead, stride)
        if n_win == 0:
            continue

        for k in range(n_win):
            s = k * stride
            # Optional NaN filter: ensure all future leads exist and are finite
            if drop_nan_targets:
                t0 = s + seq_len - 1  # last index of the input window
                targets = [flow_vec[t0 + d] for d in lead_list]
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


# =========================
# HDF slicing across years
# =========================

def _slice_across_years(h5f, paths_by_year, start_global, length):
    """
    Concatenate slices across multiple year datasets to produce a contiguous window.

    paths_by_year: list[str] (length = n_years), each dataset shape (365, H, W)
    start_global:  0-based index in the concatenated timeline
    length:        window length (L)
    """
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

def calculate_global_min_max_climate(h5_file_path):
    """
    This function calculates the global minimum and maximum values for all variables in the provided HDF5 file.
    It processes each watershed, variable, and year subset to extract the data and compute the min-max.
    NaN values are replaced with 0 before calculating min/max.

    Args:
        h5_file_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary with watershed names as keys and their corresponding min and max values for each variable.
    """
    with h5py.File(h5_file_path, 'r') as f:
        global_min_max = {}
        
        # Iterate through each watershed
        for watershed in f:
            watershed_data = {}
            
            # Iterate through each variable in the watershed
            for variable in f[watershed]:
                variable_data = []
                
                # Iterate through the years for the given variable
                for year in f[watershed][variable]:
                    # Extract the data for the year and subset
                    year_data = f[watershed][variable][year][()]
                    
                    # Replace NaN and Inf values with 0
                    year_data = np.nan_to_num(year_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Append the cleaned data for this year to the list
                    variable_data.append(year_data)
                
                # Convert list of year data into a numpy array for min-max calculation
                variable_data = np.concatenate(variable_data, axis=0)
                
                # Calculate the min and max for this variable
                min_val = np.min(variable_data)
                max_val = np.max(variable_data)
                
                # Store the min and max values
                watershed_data[variable] = {'min': min_val, 'max': max_val}
            
            # Store the results for this watershed
            global_min_max[watershed] = watershed_data

    return global_min_max

def calculate_global_min_max_streamflow(csv_path, start_year=2000):
    """
    This function calculates the global minimum and maximum values for streamflow in the provided CSV file.
    It processes each watershed, year, and replaces NaN and Inf values with 0 before computing the min-max.

    Args:
        csv_path (str): Path to the CSV file containing streamflow data.
        start_year (int): Starting year for the data (default is 2000).

    Returns:
        dict: A dictionary with watershed names as keys and their corresponding min and max values for streamflow.
    """
    # Load the CSV data into a DataFrame
    df = pd.read_csv(csv_path).copy()
    
    # Ensure the columns are treated as strings (important for consistent column names)
    df.columns = [str(c) for c in df.columns]
    
    # Drop 'date' or 'Date' column if present
    for dc in ('date', 'Date'):
        if dc in df.columns:
            df = df.drop(columns=[dc])
    
    # Initialize dictionary to store min/max values for each watershed
    global_min_max = {}

    # Get the number of years and reshape data
    n_rows = len(df)
    assert n_rows % 365 == 0, f"CSV rows ({n_rows}) not multiple of 365."
    n_years = n_rows // 365
    csv_years = [start_year + i for i in range(n_years)]

    # Iterate through each watershed
    for ws in df.columns:
        if ws in ['date', 'Date']:
            continue
        # Reshape data for the watershed into (n_years, 365) array
        series = df[ws].astype(float).to_numpy()
        arr = series.reshape(n_years, 365)  # (n_years, 365)
        
        # Replace NaN and Inf with 0 in the streamflow data
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate the min and max values for this watershed
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        # Store the min/max values for the watershed
        global_min_max[ws + "_watershed"] = {'min': min_val, 'max': max_val}

    return global_min_max

def normalize_streamflow_data(data, global_min_max_streamflow, watershed):
    """
    Normalize the streamflow data using min-max normalization.
    
    Args:
        data (np.ndarray): The streamflow data to normalize.
        global_min_max_streamflow (dict): Dictionary containing global min/max values for each watershed.
        watershed (str): The watershed name.
        
    Returns:
        np.ndarray: The normalized streamflow data.
    """
    min_val = global_min_max_streamflow[watershed]['min']
    max_val = global_min_max_streamflow[watershed]['max']
    
    # Apply min-max normalization
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return normalized_data

def normalize_climate_data(data, global_min_max, watershed, variable):
    """
    Normalize the climate data using min-max normalization.
    
    Args:
        data (np.ndarray): The climate data to normalize.
        global_min_max (dict): Dictionary containing global min/max values for each watershed and variable.
        watershed (str): The watershed name.
        variable (str): The variable (e.g., 'prcp', 'tmax', etc.) to normalize.
        
    Returns:
        np.ndarray: The normalized climate data.
    """
    min_val = global_min_max[watershed][variable]['min']
    max_val = global_min_max[watershed][variable]['max']
    
    # Apply min-max normalization
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return normalized_data

def save_min_max_to_json(global_min_max, filename):
    """
    Save global min and max values to a JSON file, converting numpy types to native Python types.
    
    Args:
        global_min_max (dict): The global min and max values.
        filename (str): The path to the JSON file.
    """
    # Convert numpy types to native Python types
    def convert_np_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)  # Convert np.float32 or np.float64 to native Python float
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)  # Convert np.int32 or np.int64 to native Python int
        return obj

    # Apply the conversion recursively to all elements in the dictionary
    global_min_max_converted = {}
    
    for ws, variables in global_min_max.items():
        watershed_data = {}
        for var, min_max_values in variables.items():
            # If the values are dictionaries (e.g., {'min': value, 'max': value})
            if isinstance(min_max_values, dict):
                watershed_data[var] = {k: convert_np_types(v) for k, v in min_max_values.items()}
            else:
                # Otherwise, it's a single numeric value
                watershed_data[var] = convert_np_types(min_max_values)
        global_min_max_converted[ws] = watershed_data
    
    # Write the converted data to the JSON file
    with open(filename, 'w') as f:
        json.dump(global_min_max_converted, f, indent=4)

        
def load_min_max_from_json(filename):
    """
    Load global min and max values from a JSON file.
    
    Args:
        filename (str): The path to the JSON file.
        
    Returns:
        dict: The global min and max values loaded from the file.
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None  # Return None if the file does not exist


# =========================
# Dataset
# =========================

class WatershedFlowDataset(Dataset):
    """
    Continuous-time windows across the *intersected* years of HDF and CSV.

    Returns for each item:
      X: torch.float32 (L, C, H, W)     — climate sequence (stacked vars)
      y: torch.float32 (K,)             — streamflow at requested lead days
      meta: dict with watershed, flow_col, start_global, seq_len, lead_days, variables, H, W
    """
    def __init__(self, h5_path, csv_path,
                 variables=None, watersheds=None,
                 seq_len=120, stride=1,
                 lead_days=None,            # NEW: int or list; if None, uses `horizon`
                 horizon=1,                 # kept for backward-compat; ignored when lead_days provided
                 start_year=2000, end_year=None,
                 drop_nan_targets=True,
                 min_max_file_climate="global_min_max.json",
                 min_max_file_streamflow="global_min_max_streamflow.json"):
        
        self.h5_path = h5_path
        self._h5 = None
        self.min_max_file_climate = min_max_file_climate
        self.min_max_file_streamflow = min_max_file_streamflow

        # 1) Load global min/max for climate data
        self.global_min_max_climate = load_min_max_from_json(self.min_max_file_climate)
        
        # 2) If not loaded from file, calculate and save
        if self.global_min_max_climate is None:
            struct = _scan_h5_ordered(h5_path, variables, watersheds,
                                      start_year=start_year, end_year=end_year)
            self.global_min_max_climate = calculate_global_min_max_climate(self.h5_path)
            save_min_max_to_json(self.global_min_max_climate, self.min_max_file_climate)
        
        # 3) Load global min/max for streamflow data
        self.global_min_max_streamflow = load_min_max_from_json(self.min_max_file_streamflow)
        
        # 4) If not loaded from file, calculate and save
        if self.global_min_max_streamflow is None:
            self.global_min_max_streamflow = calculate_global_min_max_streamflow(csv_path, start_year=start_year)
            save_min_max_to_json(self.global_min_max_streamflow, self.min_max_file_streamflow)

        # 5) CSV by-year dict (so we can stitch only common years)
        flow_by_year, _ = load_streamflow_csv_by_year(csv_path, start_year=start_year)

        # 6) HDF structure (ordered by year), limited to [start_year, end_year]
        struct = _scan_h5_ordered(h5_path, variables, watersheds,
                                  start_year=start_year, end_year=end_year)

        # 7) Build per-watershed meta aligned to common years
        per_ws_meta = _build_per_ws_meta(struct, flow_by_year,
                                         start_year=start_year, end_year=end_year)

        # 8) Normalize leads and build contiguous index
        self.lead_list = _normalize_leads(lead_days, horizon=horizon)
        self.index = _build_index_contiguous_from_meta(
            per_ws_meta, seq_len, stride, self.lead_list,
            variables=variables, drop_nan_targets=drop_nan_targets
        )

        # Keep references for __getitem__
        self.per_ws_meta = per_ws_meta
        self.ws_of = [it["watershed"] for it in self.index]

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r', libver='latest', swmr=False)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        self._ensure_open()
        it = self.index[i]
        L, H, W = it["seq_len"], it["H"], it["W"]
        C = len(it["variables"])
        s = it["start_global"]

        # Build X across years per variable
        X = np.empty((L, C, H, W), dtype=np.float32)
        for c, v in enumerate(it["variables"]):
            paths = it["var_year_paths"][v]  # ordered per-year paths for the *common years*
            X[:, c] = _slice_across_years(self._h5, paths, s, L)

            # Replace NaN and Inf with 0 in the climate data
            X[:, c] = np.nan_to_num(X[:, c], nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize the climate data using the global min/max values
            X[:, c] = normalize_climate_data(X[:, c], self.global_min_max_climate, it["watershed"], v)
        
        # y: vector for the requested lead days
        flow_vec = self.per_ws_meta[it["watershed"]]['flow_vec']
        t0 = s + L - 1
        y_vec = np.array([flow_vec[t0 + d] for d in it["lead_days"]], dtype=np.float32)  # (K,)

        # Replace NaN and Inf with 0 in the streamflow data
        y_vec = np.nan_to_num(y_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize streamflow data using the global min/max streamflow values
        y_vec = normalize_streamflow_data(y_vec, self.global_min_max_streamflow, it["watershed"])

        return {
            "seq_id": i,
            "X": torch.from_numpy(X),
            "y": torch.from_numpy(y_vec),
            "meta": {
                "watershed": it["watershed"],
                "flow_col": it["flow_col"],
                "start_global": s,
                "seq_len": L,
                "lead_days": it["lead_days"],
                "variables": it["variables"],
                "H": H, "W": W
            }
        }

# =========================
# Sampler & Collate
# =========================

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
    """X -> (B,L,C,H,W), y -> (B,K), metas list."""
    X = torch.stack([b["X"] for b in batch], 0)
    y = torch.stack([b["y"] for b in batch], 0)   # (B, K)
    metas = [b["meta"] for b in batch]
    return {"X": X, "y": y, "meta": metas}


# =========================
# Validators / Utilities
# =========================

def validate_items(dataset, n=5):
    print(f"Dataset size: {len(dataset)}")
    step = max(1, len(dataset)//n) if len(dataset) > n else 1
    checked = 0
    for i in range(0, len(dataset), step):
        sample = dataset[i]
        X, y, meta, seq_id = sample["X"], sample["y"], sample["meta"], sample["seq_id"]
        L, C, H, W = X.shape
        leads = meta['lead_days']
        y_np = y.numpy()
        preview = np.array2string(y_np[:min(5, len(y_np))], precision=3, separator=', ')
        print(f"[ITEM] seq_id={seq_id} | ws={meta['watershed']} | start_global={meta['start_global']} "
              f"| L={L} | leads={leads}")
        print(f"       X.shape={tuple(X.shape)} (L,C,H,W)  y.shape={tuple(y.shape)} values={preview}")
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
        print(f"[BATCH] size={B} | X.shape={tuple(X.shape)} | y.shape={tuple(y.shape)} "
              f"| watersheds_in_batch={dict(cnt)}")
        assert len(cnt) == 1, "Batch spans multiple watersheds (should not)."
        seen += 1
        if seen >= n_batches:
            break

def print_years_summary(h5_path, csv_path, variables=None, watersheds=None, start_year=2000, end_year=None):
    flow_by_year, _ = load_streamflow_csv_by_year(csv_path, start_year=start_year)
    struct = _scan_h5_ordered(h5_path, variables, watersheds, start_year=start_year, end_year=end_year)
    per_ws_meta = _build_per_ws_meta(struct, flow_by_year, start_year=start_year, end_year=end_year)
    print("=== Years summary (intersection) ===")
    for ws, meta in per_ws_meta.items():
        print(f"{ws}: years={meta['years_common'][0]}..{meta['years_common'][-1]} "
              f"(n={len(meta['years_common'])}), shape={meta['shape']}, vars={meta['variables']}")


# =========================
# Example usage (run this file)
# =========================

if __name__ == "__main__":
    # ---- paths
    h5_path  = "/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5"
    csv_path = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv"

    # ---- config (single watershed & single variable demo)
    variables  = None          # e.g., ["prcp"] or ["prcp","tmin"]
    watersheds = None          # list like ["4096405"] to limit
    seq_len    = 120
    stride     = 1
    # Choose exactly one of the following:
    lead_days  = 1             # contiguous leads 1..7
    # lead_days  = [1, 3, 7, 14, 30]  # or custom non-contiguous
    # horizon    = 1            # (ignored when lead_days is set)

    start_year = 2000
    end_year   = 2021          # clip to overlap with HDF/CSV

    # Optional: quick year overlap check
    print_years_summary(h5_path, csv_path, variables, watersheds, start_year, end_year)

    # ---- dataset / loader
    dataset = WatershedFlowDataset(
        h5_path=h5_path,
        csv_path=csv_path,
        variables=variables,
        watersheds=watersheds,
        seq_len=seq_len, stride=stride,
        lead_days=lead_days,            # <-- multi-lead targets
        # horizon=horizon,              # (only used if lead_days=None)
        start_year=start_year, end_year=end_year,
        drop_nan_targets=True
    )

    sampler = GroupedBatchSampler(dataset, batch_size=8, shuffle=True)
    loader  = DataLoader(dataset, batch_sampler=sampler,
                         num_workers=4, pin_memory=True,
                         collate_fn=per_ws_collate)

    # ---- sanity checks
    validate_items(dataset, n=4)
    validate_batches(loader, n_batches=2)

    # ---- example batch
    batch = next(iter(loader))
    X, y = batch["X"].float(), batch["y"].float()
    print("Example batch tensors:", X.shape, y.shape)
