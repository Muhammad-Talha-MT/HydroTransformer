# data_loader_optimized.py
# Transformation-free dataset & samplers:
#  - No transforms, no min-max scaling (use data as stored)
#  - Per-worker HDF5 handles with big rdcc caches
#  - Cache statics once per watershed (per worker)
#  - Cache (ws,var,year) climate blocks as float16
#  - Contiguous sampler for better HDF5 locality
#  - Fast per-watershed collate
#  - Leap years: if T==366, drop the last day (keep 365)

import re, h5py, numpy as np, pandas as pd, collections
from typing import Dict, Any, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, Sampler
import math

# ----------------------------- Helpers -----------------------------

def _trim_last_day_to_365(a: np.ndarray) -> np.ndarray:
    """If a is (366, H, W) or (366,), drop the last entry to get 365."""
    if a.shape[0] == 365:
        return a
    if a.shape[0] == 366:
        return a[:-1]
    raise ValueError(f"Unexpected T={a.shape[0]} (expected 365 or 366).")

def load_streamflow_csv_by_year(csv_path: str, start_year: int = 2000):
    """
    Reads daily streamflow CSV and returns:
        flow_by_year: {col: {year: np.ndarray(365,)}}
        years:        [year1, year2, ...]
    Assumes the CSV is stacked years of 365 days (common hydrology convention).
    If total rows is NOT divisible by 365, it will try a calendar/leap-aware split
    *only if* there is a date column present.
    """
    df = pd.read_csv(csv_path).copy()
    df.columns = [str(c) for c in df.columns]

    # Keep any date column around for fallback parsing only
    date_col = None
    for dc in ("date", "Date"):
        if dc in df.columns:
            date_col = dc
            break

    value_cols = [c for c in df.columns if c != date_col]
    n_rows = len(df)

    # Preferred path: fixed 365-day years (no date needed)
    if n_rows % 365 == 0:
        n_years = n_rows // 365
        years = [int(start_year) + i for i in range(n_years)]
        flow_by_year = {col: {} for col in value_cols}
        arr = df[value_cols].astype(float).to_numpy()  # (n_rows, n_cols)
        for i, y in enumerate(years):
            block = arr[i*365 : (i+1)*365, :]  # always 365
            for j, col in enumerate(value_cols):
                flow_by_year[col][y] = block[:, j].astype(np.float32, copy=False)
        return flow_by_year, years

    # Fallback: calendar-aware split if a date column exists
    if date_col is None:
        raise ValueError(
            f"CSV has {n_rows} rows, not divisible by 365 and no date column to parse."
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df["__year__"] = df[date_col].dt.year
    years = sorted(df["__year__"].unique())
    flow_by_year = {col: {} for col in value_cols}

    for y in years:
        sub = df[df["__year__"] == y][value_cols].astype(float).to_numpy()  # (T, n_cols)
        if sub.shape[0] == 366:
            sub = sub[:-1, :]
        if sub.shape[0] != 365:
            raise ValueError(f"Year {y} has {sub.shape[0]} rows after processing; expected 365.")
        for j, col in enumerate(value_cols):
            flow_by_year[col][int(y)] = sub[:, j].astype(np.float32, copy=False)

    return flow_by_year, [int(y) for y in years]

def _years_from_group(g: h5py.Group) -> List[Tuple[int, str]]:
    r = re.compile(r"^(\d{4})(?:subset\d+)?$"); out=[]
    for k in g.keys():
        m=r.match(k)
        if m: out.append((int(m.group(1)), k))
    out.sort(key=lambda x:x[0]); return out

def _scan_h5_ordered(h5_path, variables=None, watersheds=None, start_year=None, end_year=None):
    out={}
    with h5py.File(h5_path,"r") as f:
        groups=[k for k in f.keys() if k.endswith("_watershed")]
        if watersheds:
            ws_set=set(map(str,watersheds))
            groups=[g for g in groups if g.split("_")[0] in ws_set or g in ws_set]
        for ws in groups:
            var_map={}; years_sets=[]
            for var in f[ws].keys():
                if not isinstance(f[ws][var], h5py.Group): continue
                if variables and var not in variables: continue
                yrs=_years_from_group(f[ws][var])
                if not yrs: continue
                if start_year is not None or end_year is not None:
                    yrs=[(y,k) for (y,k) in yrs if (start_year is None or y>=start_year)
                                           and (end_year   is None or y<=end_year)]
                    if not yrs: continue
                var_paths=[(y,f"{ws}/{var}/{ky}") for (y,ky) in yrs]
                var_map[var]=var_paths; years_sets.append({y for y,_ in var_paths})
            if not var_map: continue
            common=None
            for s in years_sets: common=s if common is None else common.intersection(s)
            if not common: continue
            years=sorted(common)
            vars_paths_ordered={}; H=W=None
            for var,pairs in var_map.items():
                d={y:p for y,p in pairs if y in common}
                ordered=[d[y] for y in years]
                if H is None:
                    T,H,W=f[ordered[0]].shape
                    assert T in (365, 366), f"{ws}/{var} T={T}, expected 365 or 366"
                else:
                    T,h2,w2=f[ordered[0]].shape
                    assert (h2,w2)==(H,W), f"Spatial mismatch in {ws}/{var}"
                vars_paths_ordered[var]=ordered
            out[ws]={"years":years,"vars":vars_paths_ordered,"shape":(H,W)}
    return out

def default_ws_name_to_flow_col(ws_name:str)->str: return ws_name.split("_")[0]

def _normalize_leads(lead_days=None, horizon=1):
    if lead_days is None: return [int(horizon)]
    if isinstance(lead_days,int):
        assert lead_days>=1; return list(range(1,lead_days+1))
    leads=sorted({int(x) for x in lead_days}); assert leads and leads[0]>=1; return leads

def _build_per_ws_meta(per_struct, flow_by_year, start_year=None, end_year=None):
    metas={}
    for ws, meta in per_struct.items():
        col=default_ws_name_to_flow_col(ws)
        if col not in flow_by_year: continue
        ys_struct=set(meta["years"]); ys_csv=set(flow_by_year[col].keys())
        if start_year is not None:
            ys_struct={y for y in ys_struct if y>=start_year}
            ys_csv={y for y in ys_csv if y>=start_year}
        if end_year is not None:
            ys_struct={y for y in ys_struct if y<=end_year}
            ys_csv={y for y in ys_csv if y<=end_year}
        ys_common=sorted(ys_struct.intersection(ys_csv))
        if not ys_common: continue
        var_year_paths={}
        for var, ordered in meta["vars"].items():
            pby={y:p for y,p in zip(meta["years"],ordered)}
            var_year_paths[var]=[pby[y] for y in ys_common if y in pby]
        flow_vec=np.concatenate([flow_by_year[col][y] for y in ys_common], axis=0)  # each year 365 long
        metas[ws]={"watershed":ws,"years_common":ys_common,"shape":meta["shape"],
                   "variables":list(var_year_paths.keys()),"var_year_paths":var_year_paths,
                   "flow_col":col,"flow_vec":flow_vec}
    return metas

def _num_windows_contiguous(total_days, seq_len, max_lead, stride):
    max_start=total_days - seq_len - max_lead
    if max_start<0: return 0
    return (max_start//stride)+1

def _build_index_contiguous_from_meta(per_ws_meta, seq_len, stride, lead_days, variables=None, drop_nan_targets=True):
    lead_list=_normalize_leads(lead_days, horizon=1)
    max_lead=max(lead_list)-1
    index=[]
    for ws, meta in per_ws_meta.items():
        vars_here=variables if variables else meta["variables"]
        vars_here=[v for v in vars_here if v in meta["var_year_paths"]]
        if not vars_here: continue
        H,W=meta["shape"]; flow_vec=meta["flow_vec"]; total_days=flow_vec.shape[0]
        n_win=_num_windows_contiguous(total_days, seq_len, max_lead, stride)
        if n_win==0: continue
        for k in range(n_win):
            s=k*stride
            if drop_nan_targets:
                t0=s+seq_len-1
                targets=[flow_vec[t0+(d-1)] for d in lead_list]
                if any((np.isnan(v) or np.isinf(v)) for v in targets): continue
            index.append({"watershed":ws,"flow_col":meta["flow_col"],"start_global":s,"seq_len":seq_len,
                          "H":H,"W":W,"variables":vars_here,"var_year_paths":{v:meta["var_year_paths"][v] for v in vars_here},
                          "lead_days":lead_list})
    return index

def _pick_first_dataset(grp: h5py.Group, prefer_suffix=None):
    if prefer_suffix is not None:
        for k in grp.keys():
            obj=grp[k]
            if isinstance(obj,h5py.Dataset) and k.lower().endswith(prefer_suffix.lower()): return obj
    for k in grp.keys():
        obj=grp[k]
        if isinstance(obj,h5py.Dataset): return obj
    return None

# ----------------------------- Dataset -----------------------------

class WatershedFlowDataset(Dataset):
    def __init__(self, h5_path, csv_path, static_h5,
                 variables=None, watersheds=None,
                 seq_len=120, stride=1,
                 lead_days=None, horizon=1,
                 start_year=2000, end_year=None,
                 drop_nan_targets=True,
                 year_cache_bytes=1_200_000_000):
        # lazy handles & caches
        self.h5_path=h5_path; self._h5=None
        self.static_h5_path=static_h5; self._static_h5=None
        self._static_cache={}
        self._year_cache=collections.OrderedDict()
        self._year_cache_used=0; self._year_cache_bytes=int(year_cache_bytes)

        # streamflow by-year (each year 365 after trimming 366th if present)
        flow_by_year,_=load_streamflow_csv_by_year(csv_path, start_year=start_year)
        # climate structure
        struct=_scan_h5_ordered(h5_path, variables, watersheds, start_year=start_year, end_year=end_year)
        # per-WS meta
        per_ws_meta=_build_per_ws_meta(struct, flow_by_year, start_year=start_year, end_year=end_year)

        # index
        self.lead_list=_normalize_leads(lead_days, horizon=1)
        self.index=_build_index_contiguous_from_meta(per_ws_meta, seq_len, stride, self.lead_list,
                                                     variables=variables, drop_nan_targets=drop_nan_targets)
        self.per_ws_meta=per_ws_meta
        self.ws_of=[it["watershed"] for it in self.index]

    def _ensure_open(self):
        if self._h5 is None:
            self._h5=h5py.File(self.h5_path,"r", libver="latest", swmr=False,
                               rdcc_nslots=1_000_003, rdcc_nbytes=512*1024**2, rdcc_w0=0.75)
        if self._static_h5 is None:
            self._static_h5=h5py.File(self.static_h5_path,"r",
                                      rdcc_nslots=200_003, rdcc_nbytes=128*1024**2, rdcc_w0=0.75)

    def _static_group_for_ws(self, ws_name):
        g=self._static_h5.get(ws_name,None)
        if g is not None: return g
        return self._static_h5.get(ws_name.split("_")[0], None)

    def _get_static_cached(self, ws_name: str):
        st=self._static_cache.get(ws_name)
        if st is not None: return st
        g=self._static_group_for_ws(ws_name)
        DEM=awc=fc=soil=None
        if g is not None:
            def _read_plain(ds):
                a=ds[()]
                if a.ndim==3 and a.shape[0]==1:
                    a=a[0]
                return a.astype(np.float32, copy=False)

            if "DEM_clips" in g:
                ds=_pick_first_dataset(g["DEM_clips"])
                if ds is not None:
                    DEM=torch.from_numpy(_read_plain(ds))

            if "awc_clips" in g:
                ds=_pick_first_dataset(g["awc_clips"], prefer_suffix="_awc")
                if ds is not None:
                    awc=torch.from_numpy(_read_plain(ds))

            if "fc_clips" in g:
                ds=_pick_first_dataset(g["fc_clips"], prefer_suffix="_fc")
                if ds is not None:
                    fc=torch.from_numpy(_read_plain(ds))

            if "soil_clips" in g:
                soil_grp=g["soil_clips"]; chans=[]
                for sub in ("clay","sand","silt"):
                    ds=None
                    for k in soil_grp.keys():
                        obj=soil_grp[k]
                        if isinstance(obj,h5py.Dataset) and k.lower().endswith(f"_{sub}"):
                            ds=obj; break
                    if ds is None: continue
                    chans.append(_read_plain(ds))
                if chans:
                    Hc=min(c.shape[0] for c in chans); Wc=min(c.shape[1] for c in chans)
                    soil=torch.from_numpy(np.stack([c[:Hc,:Wc] for c in chans], axis=0))
        self._static_cache[ws_name]=(DEM,awc,fc,soil)
        if len(self._static_cache)>256:
            self._static_cache.pop(next(iter(self._static_cache)))
        return self._static_cache[ws_name]

    def _year_cache_evict_if_needed(self):
        while self._year_cache and self._year_cache_used>self._year_cache_bytes:
            k,arr=self._year_cache.popitem(last=False)
            self._year_cache_used-=arr.nbytes

    def _get_year_block(self, ws_name:str, var:str, year_idx:int, path:str)->np.ndarray:
        """
        Returns a cached (365,H,W) float16 block (leap-day trimmed).
        No transforms, no min-max — uses stored values.
        """
        key=(ws_name,var,year_idx)
        arr=self._year_cache.get(key)
        if arr is not None:
            self._year_cache.move_to_end(key, last=True); return arr

        ds=self._h5[path]; a=ds[()]  # (T,H,W)
        if a.shape[0] == 366:
            a = a[:-1]
        a=a.astype(np.float32, copy=False, order="C")

        a16=a.astype(np.float16, copy=False)
        self._year_cache[key]=a16; self._year_cache_used+=a16.nbytes
        self._year_cache.move_to_end(key, last=True); self._year_cache_evict_if_needed()
        return a16

    def __len__(self): return len(self.index)

    def __getitem__(self, i:int):
        self._ensure_open()
        it=self.index[i]
        L,H,W=it["seq_len"],it["H"],it["W"]; C=len(it["variables"]); s=it["start_global"]; ws=it["watershed"]
        X=np.empty((L,C,H,W), dtype=np.float16)
        for c,v in enumerate(it["variables"]):
            paths=it["var_year_paths"][v]
            Xi=np.empty((L,H,W), dtype=np.float16)
            remaining=L; pos=0; t=s
            while remaining>0:
                yr=t//365; off=t%365
                year_block=self._get_year_block(ws, v, yr, paths[yr])  # always 365 after trimming
                take=min(365-off, remaining)
                Xi[pos:pos+take]=year_block[off:off+take]
                t+=take; pos+=take; remaining-=take
            X[:,c]=Xi

        DEM,awc,fc,soil=self._get_static_cached(ws)
        flow_vec=self.per_ws_meta[ws]["flow_vec"]
        t0=s+L-1
        y=np.array([flow_vec[t0+(d-1)] for d in it["lead_days"]], dtype=np.float32)

        return {
            "seq_id": i,
            "X": torch.from_numpy(X),
            "DEM": DEM, "awc":awc, "fc":fc, "soil":soil,
            "y": torch.from_numpy(y),
            "meta": {"watershed":ws,"flow_col":it["flow_col"],"start_global":s,"seq_len":L,
                     "lead_days":it["lead_days"],"variables":it["variables"],"H":H,"W":W}
        }

    def __del__(self):
        try:
            if self._h5: self._h5.close()
        except Exception: pass
        try:
            if self._static_h5: self._static_h5.close()
        except Exception: pass

# ----------------------------- Samplers -----------------------------

class GroupedContiguousSampler(Sampler):
    def __init__(self, dataset: WatershedFlowDataset, batch_size=8, block_len=256, shuffle=True):
        self.batch_size=int(batch_size); self.block_len=int(block_len); self.shuffle=bool(shuffle)
        buckets={}
        for i,ws in enumerate(dataset.ws_of):
            buckets.setdefault(ws,[]).append(i)
        self.blocks=[]
        rng=np.random.default_rng()
        for _, idxs in buckets.items():
            blocks=[idxs[k:k+self.block_len] for k in range(0,len(idxs), self.block_len)]
            if self.shuffle: rng.shuffle(blocks)
            self.blocks.extend(blocks)
        if self.shuffle: rng.shuffle(self.blocks)

    def __iter__(self):
        for block in self.blocks:
            for k in range(0, len(block), self.batch_size):
                yield block[k:k+self.batch_size]

    def __len__(self):
        total=0
        for block in self.blocks:
            total += (len(block)+self.batch_size-1)//self.batch_size
        return total

GroupedBatchSampler = GroupedContiguousSampler  # alias

def per_ws_collate_optimized(batch):
    X=torch.stack([b["X"] for b in batch],0)   # (B,L,C,H,W) float16
    y=torch.stack([b["y"] for b in batch],0)   # (B,K) float32
    metas=[b["meta"] for b in batch]
    b0=batch[0]
    return {"X":X,"y":y,"meta":metas,
            "DEM":b0.get("DEM"),"awc":b0.get("awc"),"fc":b0.get("fc"),"soil":b0.get("soil")}

class DistributedGroupedBatchSampler(Sampler):
    """
    DDP-aware version of GroupedContiguousSampler with optional drop_last.
    - batch_size is per-rank (global batch = batch_size * world_size).
    - Call set_epoch(epoch) each epoch for consistent shuffling across ranks.
    - If drop_last=True, the global list of batches is truncated so that
      len(batches) % world_size == 0, ensuring identical steps per rank.
    """
    def __init__(
        self,
        dataset,
        batch_size=8,
        block_len=256,
        shuffle=True,
        rank=0,
        world_size=1,
        seed=123,
        drop_last=False,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.block_len = int(block_len)
        self.shuffle = bool(shuffle)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        # Build buckets per watershed (deterministic order)
        buckets = {}
        for i, ws in enumerate(dataset.ws_of):
            buckets.setdefault(ws, []).append(i)

        # Pre-split each watershed sequence into contiguous blocks
        self._all_blocks = []
        for _, idxs in buckets.items():
            blocks = [idxs[k:k + self.block_len] for k in range(0, len(idxs), self.block_len)]
            self._all_blocks.extend(blocks)

        # Precompute the total number of batches (rank-agnostic) for __len__
        self._total_batches_all_ranks = self._compute_total_batches_all_ranks()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _compute_total_batches_all_ranks(self) -> int:
        total = 0
        for block in self._all_blocks:
            blk_len = len(block)
            if self.drop_last:
                total += blk_len // self.batch_size
            else:
                total += (blk_len + self.batch_size - 1) // self.batch_size  # ceil
        if self.drop_last:
            total = (total // self.world_size) * self.world_size
        return total

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)

        blocks = list(self._all_blocks)
        if self.shuffle:
            rng.shuffle(blocks)

        batches = []
        for block in blocks:
            for k in range(0, len(block), self.batch_size):
                batch = block[k:k + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        if self.drop_last and len(batches) % self.world_size != 0:
            keep = (len(batches) // self.world_size) * self.world_size
            batches = batches[:keep]

        local_batches = batches[self.rank::self.world_size]

        for b in local_batches:
            yield b

    def __len__(self):
        N = self._total_batches_all_ranks
        if self.drop_last:
            return N // self.world_size
        else:
            if N <= self.rank:
                return 0
            return math.ceil((N - self.rank) / self.world_size)

