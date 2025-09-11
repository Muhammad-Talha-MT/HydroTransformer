# From Raw Files → Ready-to-Train Batches

This document lays out the end-to-end mental model for the dataset pipeline: how we go from raw CSV/HDF5 files to batched tensors that a model can consume.

---

## What the code builds

**Goal:** For each watershed, generate many training examples where:

- **X** = last `L` days of climate maps, shaped `(L, C, H, W)`
- **y** = streamflow for **multiple future leads**, shaped `(K,)` where `K = len(lead_days)`

**Continuous time:** Sequences may cross year boundaries (e.g., Dec 30 → Jan 1), so the pipeline uses the full contiguous timeline.

---

## Data ingestion

### 1) CSV (streamflow) → per-year dict

`load_streamflow_csv_by_year(csv_path, start_year=2000)` reads the CSV and reshapes each watershed column into `(n_years, 365)`. It returns:

- `flow_by_year[ws_id][year] -> (365,)`
- `csv_years` (sorted list of years inferred from rows)

**Assumptions:**  
- Header has watershed IDs as columns (plus optional `date` column, dropped).
- Exactly 365 rows per year (leap days removed beforehand).

---

### 2) HDF5 (climate) → ordered per-year paths

`_scan_h5_ordered(h5, variables, watersheds, start_year, end_year)` walks the HDF5 file and:

- Keeps datasets like `4096405_watershed/var/2000subsetX` (one dataset per year).
- Filters to the provided `variables`, `watersheds`, and the year range `[start_year, end_year]`.
- **Intersects years across all chosen variables** (only years present for **every** variable are kept).
- Stores, per watershed:
  - `years` (sorted list kept after intersection)
  - `vars[var]` = list of dataset **paths**, one per year, ordered to match `years`
  - `shape = (H, W)` taken from the first dataset

---

## Aligning climate with streamflow

### 3) Per-watershed metadata (“meta”)

`_build_per_ws_meta(h5, scan_result, flow_by_year)`:

- Intersects HDF5 `years` with CSV `csv_years` → `years_common`.
- Reorders each var’s dataset paths to match `years_common`.
- **Concatenates streamflow** across `years_common` into a single vector:
  - `flow_vec` shape = `365 * len(years_common)`

This metadata is the contract the indexer/runtime slicing relies on.

---

## Window indexing (what samples exist)

### 4) Multi-lead index

`lead_days` may be:

- An integer (e.g., `lead_days=7`) → contiguous leads `[1, 2, …, 7]`
- A list (e.g., `lead_days=[1, 3, 7, 14, 30]`) → custom leads

`_normalize_leads(lead_days)` returns a sorted list `lead_list`.

`_build_index_contiguous_from_meta(meta, seq_len=L, stride=1, drop_nan_targets=True)` creates **one index item per valid start**:

- `start_global = s` (0-based) within the concatenated timeline.
- Requires room for the largest lead: `max_lead = max(lead_days)`.
- If `drop_nan_targets=True`, drops windows where any target lead is NaN/Inf.
- Each index item stores:
  - `ws_id`, `seq_len=L`, `(H, W)`, list of `variables`
  - `paths_by_year` per variable (ordered per `years_common`)
  - `lead_days`

**Dataset size per watershed (before NaN drops):**

\[
N \;=\; \left\lfloor \frac{T - L - \max(\text{lead\_days})}{\text{stride}} \right\rfloor + 1
\quad\text{where}\quad
T = 365 \times |\text{years\_common}|
\]

---

## Turning an index item into `(X, y)`

### 5) `__getitem__` (runtime slicing)

Given an index item:

- **Build `X`**  
  For each variable, call:
