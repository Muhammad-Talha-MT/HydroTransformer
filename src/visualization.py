import os, re, json, h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt
from statistics import NormalDist

# ---------- Helpers (match your loader's logic) ----------
def _years_from_group_names(g):
    yr_re = re.compile(r"^(\d{4})subset\d+$")
    pairs = []
    for k in g.keys():
        m = yr_re.match(k)
        if m:
            pairs.append((int(m.group(1)), k))
    pairs.sort(key=lambda x: x[0])
    return pairs

def _transform_array(arr, kind=None):
    if kind in (None, "identity"): return arr.astype(np.float32)
    arr = arr.astype(np.float32)
    if kind == "log1p":
        arr = np.where(arr < 0.0, 0.0, arr)
        return np.log1p(arr)
    if kind == "asinh": return np.arcsinh(arr)
    if kind == "sqrt":
        arr = np.where(arr < 0.0, 0.0, arr)
        return np.sqrt(arr)
    raise ValueError(f"Unknown transform: {kind}")

def _normalize(arr, mn, mx):
    den = (mx - mn) if (mx - mn) != 0 else 1.0
    return (arr - mn) / den

def _ecdf(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.array([0.0]), np.array([0.0])
    xs = np.sort(x)
    ys = (np.arange(1, xs.size+1)) / xs.size
    return xs, ys

def _skewness(x):
    x = x[np.isfinite(x)]
    if x.size < 2: return np.nan
    m = x.mean(); s = x.std()
    if s == 0: return 0.0
    return np.mean(((x - m)/s)**3)

def _pct_zeros(x, eps=0.0):
    x = x[np.isfinite(x)]
    if x.size == 0: return 0.0
    return 100.0 * np.mean(x <= eps)

def suggest_transform(values):
    """Heuristic to offer a starting transform."""
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if v.size < 100: return "identity"
    signed = (np.min(v) < 0)
    frac_zero = np.mean(v == 0)
    sk = _skewness(v)
    if not signed and (frac_zero > 0.01 or sk > 1.0):
        return "log1p"
    if signed and abs(sk) > 1.0:
        return "asinh"
    return "identity"

def _qqplot_against_normal(ax, x, title):
    x = x[np.isfinite(x)]
    if x.size < 50:
        ax.text(0.5, 0.5, "Too few points", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    x = (x - x.mean()) / (x.std() if x.std() != 0 else 1.0)
    n = x.size
    p = (np.arange(1, n+1) - 0.5) / n
    q_theor = np.array([NormalDist().inv_cdf(pi) for pi in p])
    q_sample = np.sort(x)
    ax.scatter(q_theor, q_sample, s=6, alpha=0.6)
    lim = np.nanpercentile(np.concatenate([q_theor, q_sample]), [1, 99])
    ax.plot(lim, lim, lw=1.0)
    ax.set_title(title); ax.set_xlabel("Theoretical N(0,1) quantiles"); ax.set_ylabel("Sample quantiles")
    ax.grid(alpha=0.3)

# ---------- GLOBAL-first min–max helpers ----------
def _load_minmax(json_path):
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        return json.load(f)

def _get_climate_minmax(gmm, watershed_key, variable):
    """
    Prefer GLOBAL per-variable stats: gmm['GLOBAL'][var]{min,max}
    Fallbacks:
      - gmm[watershed_key][variable]
      - gmm[watershed_id_only][variable]
    """
    if gmm is None: return (None, None)
    if 'GLOBAL' in gmm and isinstance(gmm['GLOBAL'], dict):
        mm = gmm['GLOBAL'].get(variable)
        if mm and 'min' in mm and 'max' in mm:
            return float(mm['min']), float(mm['max'])
    # legacy per-watershed
    for key in (watershed_key, watershed_key.split('_')[0]):
        mm_ws = gmm.get(key, {})
        mm = mm_ws.get(variable)
        if mm and 'min' in mm and 'max' in mm:
            return float(mm['min']), float(mm['max'])
    return (None, None)

def _get_flow_minmax(gmm, watershed_id):
    """
    Prefer GLOBAL flow stats: gmm['GLOBAL']{min,max}
    Fallback: per-watershed key like '<id>_watershed'
    """
    if gmm is None: return (None, None)
    if 'GLOBAL' in gmm and isinstance(gmm['GLOBAL'], dict) and 'min' in gmm['GLOBAL']:
        mm = gmm['GLOBAL']
        return float(mm['min']), float(mm['max'])
    ws_key = f"{watershed_id}_watershed"
    mm = gmm.get(ws_key)
    if mm and 'min' in mm and 'max' in mm:
        return float(mm['min']), float(mm['max'])
    return (None, None)

# ---------- Sampling from your data ----------
def _collect_climate_values(h5_path, watershed, variable,
                            years=None, time_stride=5, space_stride=5, max_years=None):
    """Downsample in time and space to keep it light."""
    vals = []
    with h5py.File(h5_path, 'r') as f:
        g = f[watershed][variable]
        yr_pairs = _years_from_group_names(g)
        if years is not None:
            years_set = set(years)
            yr_pairs = [(y,k) for (y,k) in yr_pairs if y in years_set]
        if max_years is not None:
            yr_pairs = yr_pairs[:max_years]
        for _, ky in yr_pairs:
            ds = g[ky]           # (365, H, W)
            take = ds[::time_stride, ::space_stride, ::space_stride]
            vals.append(take.reshape(-1))
    if not vals:
        return np.array([])
    x = np.concatenate(vals, axis=0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def _collect_streamflow_values(csv_path, watershed_id, start_year=2000, years=None):
    """Returns raw daily flows for a single gauge as a flat vector."""
    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c) for c in df.columns]
    for dc in ("date","Date"):
        if dc in df.columns: df = df.drop(columns=[dc])
    if watershed_id not in df.columns:
        raise KeyError(f"{watershed_id} not found in CSV columns.")
    x = df[watershed_id].astype(float).to_numpy()
    n = x.shape[0]
    assert n % 365 == 0, "CSV must be whole years"
    n_years = n // 365
    if years is not None:
        keep = []
        for y in years:
            idx = (y - start_year)
            if 0 <= idx < n_years:
                keep.append(x[idx*365:(idx+1)*365])
        x = np.concatenate(keep, axis=0) if keep else np.array([])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x

# ---------- Core plotting ----------
def _visualize_base(x_raw, transform_kind, norm_min=None, norm_max=None,
                    title_prefix="", bins=80, show_qq=True,
                    save_path=None, dpi=300, show=False):
    """
    Raw vs Transformed vs Normalized with ECDFs (+ optional QQ).
    Prefer passing GLOBAL norm_min/max; falls back to data min/max if None.
    """
    x_raw = x_raw[np.isfinite(x_raw)]
    if x_raw.size == 0:
        print("No data to visualize.")
        return

    x_tr  = _transform_array(x_raw, transform_kind)
    mn = x_tr.min() if norm_min is None else norm_min
    mx = x_tr.max() if norm_max is None else norm_max
    x_nm = _normalize(x_tr, mn, mx)

    def _annot(ax, v, label, extra_zero=None):
        m, s, sk = float(np.mean(v)), float(np.std(v)), float(_skewness(v))
        txt = f"{label}\nμ={m:.3g}, σ={s:.3g}, skew={sk:.3g}"
        if extra_zero is not None:
            txt += f"\n% zeros={extra_zero:.2f}%"
        ax.text(
            0.02, 0.98, txt,
            va="top", ha="left",               # ← fixed: use a single valid value
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
            fontsize=9,
        )

    rows = 2 if show_qq else 1
    fig, axes = plt.subplots(rows, 3, figsize=(15, 6 if show_qq else 4), constrained_layout=True)
    if getattr(axes, "ndim", 1) == 1:
        axes = np.array([axes])

    # Row 0: hist + ECDF
    for j, (dat, label) in enumerate([(x_raw,"Raw"), (x_tr,"Transformed"), (x_nm,"Normalized [0,1]")]):
        axh = axes[0, j]
        axh.hist(dat, bins=bins, alpha=0.85)
        axh.set_title(f"{title_prefix} — {label}")
        axh.grid(alpha=0.3)
        z = _pct_zeros(x_raw) if label == "Raw" else None
        _annot(axh, dat, label, extra_zero=z)
        xs, ys = _ecdf(dat)
        ax2 = axh.twinx(); ax2.plot(xs, ys, lw=1.25, alpha=0.9); ax2.set_ylabel("ECDF")

    # Row 1: QQ
    if show_qq:
        _qqplot_against_normal(axes[1,0], x_raw, "QQ vs Normal (Raw)")
        _qqplot_against_normal(axes[1,1], x_tr,  "QQ vs Normal (Transformed)")
        _qqplot_against_normal(axes[1,2], x_nm,  "QQ vs Normal (Normalized)")

    # Save / show / close
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[saved] {save_path}")
    if show:
        plt.show()
    plt.close(fig)

# ---------- Public visualization APIs ----------
def visualize_climate_distribution(h5_path, watershed, variable,
                                   transform_kind="log1p",
                                   minmax_json_path=None,
                                   years=None, time_stride=5, space_stride=5,
                                   bins=80, show_qq=True,
                                   save_path=None, dpi=300, show=False):
    """
    watershed: HDF group key, e.g. '4159900_watershed'
    minmax_json_path: expects GLOBAL per-variable stats (preferred).
    """
    x_raw = _collect_climate_values(h5_path, watershed, variable, years, time_stride, space_stride)

    norm_min = norm_max = None
    gmm = _load_minmax(minmax_json_path)
    if gmm is not None:
        norm_min, norm_max = _get_climate_minmax(gmm, watershed_key=watershed, variable=variable)

    title = f"{watershed}:{variable} ({transform_kind})"
    _visualize_base(x_raw, transform_kind, norm_min, norm_max, title,
                    bins=bins, show_qq=show_qq, save_path=save_path, dpi=dpi, show=show)

def visualize_streamflow_distribution(csv_path, watershed_id,
                                      transform_kind="log1p",
                                      minmax_json_path=None,
                                      start_year=2000, years=None,
                                      bins=80, show_qq=True,
                                      save_path=None, dpi=300, show=False):
    """
    watershed_id: CSV column name like '4159900' (no '_watershed' suffix).
    minmax_json_path: expects GLOBAL flow stats (preferred).
    """
    x_raw = _collect_streamflow_values(csv_path, watershed_id, start_year, years)

    norm_min = norm_max = None
    gmm = _load_minmax(minmax_json_path)
    if gmm is not None:
        norm_min, norm_max = _get_flow_minmax(gmm, watershed_id)

    title = f"{watershed_id}:streamflow ({transform_kind})"
    _visualize_base(x_raw, transform_kind, norm_min, norm_max, title,
                    bins=bins, show_qq=show_qq, save_path=save_path, dpi=dpi, show=show)

# ---------- Script entry ----------
if __name__ == "__main__":
    # Climate variable (e.g., precip) – zero-inflated, so log1p:
    visualize_climate_distribution(
        h5_path="/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5",
        watershed="4159900_watershed",
        variable="prcp",
        transform_kind="log1p",
        minmax_json_path="log10__CLIMATE__TRAIN_GLOBAL.json",  # with {'GLOBAL': {var: {min,max}}}
        save_path="plots/climate/4159900_prcp_log1p.png",
        dpi=300, show=False
    )

    # Streamflow plot (GLOBAL min–max preferred)
    visualize_streamflow_distribution(
        csv_path="/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv",
        watershed_id="4159900",
        transform_kind="log1p",
        minmax_json_path="log10__FLOW__TRAIN_GLOBAL.json",     # with {'GLOBAL': {min,max}}
        save_path="plots/flow/4159900_flow_log1p.pdf",
        dpi=300, show=False
    )

    watersheds = [
        4096405, 4096515, 4097500, 4097540, 4099000, 4101500, 4101800, 4102500,
        4102700, 4104945, 4105000, 4105500, 4105700, 4106000, 4108600, 4108800,
        4109000, 4112000, 4112500, 4113000, 4114000, 4115000, 4115265, 4116000,
        4117500, 4118500, 4121300, 4121500, 4121944, 4121970, 4122100, 4122200,
        4122500, 4124200, 4124500, 4125550, 4126740, 4126970, 4127800, 4142000,
        4144500, 4146000, 4146063, 4147500, 4148140, 4148500, 4151500, 4152238,
        4154000, 4157005, 4159492, 4159900, 4160600, 4163400, 4164100, 4164300,
        4166500, 4167000, 4175600, 4176000, 4176500
    ]
    watersheds_list = [str(item) for item in watersheds]

    # Batch save: multiple vars/watersheds (GLOBAL min–max preferred)
    for ws in watersheds_list:
        # Climate
        visualize_climate_distribution(
            h5_path="/data/HydroTransformer/daymet/daymet_watersheds_clipped.h5",
            watershed=ws+"_watershed",
            variable="prcp",
            transform_kind="log1p",
            minmax_json_path="log10__CLIMATE__TRAIN_GLOBAL.json",
            save_path=f"plots/batch-log10/{ws}_prcp_log1p.png",
            show=False
        )
        # Streamflow
        visualize_streamflow_distribution(
            csv_path="/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/HydroTransformer_Streamflow.csv",
            watershed_id=ws,
            transform_kind="log1p",
            minmax_json_path="log10__FLOW__TRAIN_GLOBAL.json",
            save_path=f"plots/flow-log10/{ws}_flow_log1p.png",
            dpi=300, show=False
        )
