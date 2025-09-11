#!/usr/bin/env python3
import os
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
from pyproj import CRS
from rasterio.features import rasterize

# ───────── CONFIGURATION ─────────
DAYMET_DIR = Path("/data/HydroTransformer/")
SHAPEFILE_DIR = Path("/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/watershed_shapefiles")
OUTPUT_H5 = Path("/data/HydroTransformer/daymet_watersheds_clipped.h5")
REGION = 'na'  # 'na', 'hi', or 'pr'
VARIABLES = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
YEARS = range(1980, 2022)
LCC_PROJ4 = (
    "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 "
    "+lon_0=-100 +x_0=0 +y_0=0 +units=km +ellps=WGS84 +no_defs"
)
LCC_CRS = CRS.from_proj4(LCC_PROJ4)
# ───────────────────────────────────

def prepare_window_and_mask(shapefile_path: Path, sample_nc: Path, var_name: str):
    """
    Load watershed, reproject to LCC, compute bbox on Daymet grid,
    return (x_slice, y_slice), and a boolean mask (rows, cols) on that window.
    """
    # 1) Watershed to LCC
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(LCC_CRS)
    geom = gdf.geometry.union_all()  # handle multi-polygons robustly
    geom_json = [mapping(geom)]

    # 2) Open one dataset to get grid & coords
    #    Chunk to let dask avoid loading full array; we only need coords.
    with xr.open_dataset(sample_nc, chunks={"time": 32, "y": 512, "x": 512}) as ds:
        da = ds[var_name]
        # Ensure we know x,y are the spatial dims
        assert 'x' in da.dims and 'y' in da.dims, "Expected x,y dims in Daymet."
        x = da['x'].values
        y = da['y'].values  # Daymet often descending

        # 3) Compute bbox in grid coordinates (min/max in same units as x,y)
        minx, miny, maxx, maxy = gdf.total_bounds
        # Build index slices by finding nearest indices in the coordinate arrays
        def idx_slice(coord, vmin, vmax, descending=False):
            if descending:
                i0 = np.searchsorted(coord[::-1], vmin, side='left')
                i1 = np.searchsorted(coord[::-1], vmax, side='right') - 1
                # map back to original indices
                n = coord.size
                j0 = n - 1 - i1
                j1 = n - 1 - i0
                return slice(max(0, j0), min(n, j1 + 1)), True
            else:
                i0 = np.searchsorted(coord, vmin, side='left')
                i1 = np.searchsorted(coord, vmax, side='right') - 1
                return slice(max(0, i0), min(coord.size, i1 + 1)), False

        x_sl, _ = idx_slice(x, minx, maxx, descending=False)
        # y is often descending in geophysical grids
        y_desc = (y[0] > y[-1])
        y_sl, _ = idx_slice(y, miny, maxy, descending=y_desc)

        # 4) Create a mask on the window grid by rasterizing the polygon
        xw = x[x_sl]
        yw = y[y_sl]
        # Build transform for this window (assumes regular spacing)
        dx = float(np.abs(xw[1] - xw[0])) if xw.size > 1 else 1.0
        dy = float(np.abs(yw[1] - yw[0])) if yw.size > 1 else 1.0
        # Upper-left corner depends on axis order; for rasterize we need affine transform:
        # Transform from (col,row) -> (x,y):
        #    X = x0 + col*dx
        #    Y = y0 - row*dy  if y is descending (common)
        x0 = xw[0]
        y0 = yw[0]
        from affine import Affine
        if y_desc:
            transform = Affine(dx, 0, x0, 0, -dy, y0)
        else:
            transform = Affine(dx, 0, x0, 0, dy, y0)

        out_shape = (yw.size, xw.size)
        mask = rasterize(
            shapes=geom_json,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype='uint8'
        ).astype(bool)

    return x_sl, y_sl, mask, y_desc

def write_var_year(nc_file: Path, var_name: str, h5_group: h5py.Group,
                   x_sl: slice, y_sl: slice, mask: np.ndarray, y_desc: bool):
    # Open lazily, load only window
    with xr.open_dataset(nc_file, chunks={"time": 32, "y": 512, "x": 512}) as ds:
        da = ds[var_name].isel(x=x_sl, y=y_sl)
        # Fetch into memory as numpy (only the needed block)
        arr = da.values  # shape: (time, rows, cols)

        # If y is descending and you want south→north, flip here
        if y_desc:
            arr = arr[:, ::-1, :]
            mask_use = mask[::-1, :]
        else:
            mask_use = mask

        # Apply mask across time (vectorized)
        masked = np.where(mask_use[None, :, :], arr, np.nan)

        # Dataset name is the year (string)
        year = nc_file.stem.split('_')[-1]  # e.g., "2000subset6"
        var_grp = h5_group.require_group(var_name)
        dset = var_grp.create_dataset(
            year,
            data=masked,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )

        # Store dates as a tiny dataset once per (var/year), not a big attribute
        # This is safer than attributes for large arrays.
        time_vals = ds['time'].values
        date_str = np.array([np.datetime_as_string(t, unit='D') for t in time_vals], dtype='S10')
        # Put dates here: /<watershed>/<var>/_dates/<year>
        date_grp = var_grp.require_group("_dates")
        date_grp.create_dataset(year, data=date_str, compression="gzip", compression_opts=1)

def main():
    # Safer HDF5 options; libver="latest" improves robustness for big files
    OUTPUT_H5.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_H5.exists():
        OUTPUT_H5.unlink()  # don't append to a corrupted file

    with h5py.File(OUTPUT_H5, "w", libver="latest") as hf:
        for shp in SHAPEFILE_DIR.rglob("*.shp"):
            ws_name = shp.stem
            print(f"Processing watershed: {ws_name}")
            ws_grp = hf.create_group(ws_name)

            for var in VARIABLES:
                var_dir = DAYMET_DIR / f"CONUS_{var}_nc_subset6"
                if not var_dir.is_dir():
                    print(f"  ⚠️ Dir not found for {var}: {var_dir}, skipping")
                    continue

                # pick one sample year that exists to build window+mask once
                sample_nc = None
                for y in YEARS:
                    cand = var_dir / f"{var}_{y}subset6.nc"
                    if cand.is_file():
                        sample_nc = cand
                        break
                if sample_nc is None:
                    print(f"  ⚠️ No files for {var}, skipping")
                    continue

                # Build window+mask ONCE per watershed per variable
                x_sl, y_sl, mask, y_desc = prepare_window_and_mask(shp, sample_nc, var)

                for year in YEARS:
                    nc_file = var_dir / f"{var}_{year}subset6.nc"
                    if nc_file.is_file():
                        write_var_year(nc_file, var, ws_grp, x_sl, y_sl, mask, y_desc)

            hf.flush()  # flush after each watershed to reduce corruption risk
            print(f"✅ Completed watershed: {ws_name}\n")

    print(f"All watersheds written to {OUTPUT_H5}")

if __name__ == "__main__":
    main()
