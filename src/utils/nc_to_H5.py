#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import h5py
from pyproj import CRS, Transformer
import geopandas as gpd
from pathlib import Path
import rioxarray  # patches xarray with .rio methods
import matplotlib.pyplot as plt
import io
from PIL import Image


def print_extent_and_region(nc_path: str):
    # 1. Open dataset and get native x/y bounds
    ds = xr.open_dataset(nc_path)
    x_min, x_max = ds.x.min().item(), ds.x.max().item()
    y_min, y_max = ds.y.min().item(), ds.y.max().item()
    units = ds.x.attrs.get("units", "unknown")

    print(f"LCC (native) x range: {x_min:.2f} → {x_max:.2f} {units}")
    print(f"LCC (native) y range: {y_min:.2f} → {y_max:.2f} {units}\n")

    # 2. Define your Daymet‐style LCC CRS (units=km)
    lcc_def = (
        "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=km +no_defs"
    )
    crs_lcc   = CRS.from_proj4(lcc_def)
    crs_wgs84 = CRS.from_epsg(4326)
    crs_nad83 = CRS.from_epsg(4269)

    # 3. Build transformers
    tf_wgs84 = Transformer.from_crs(crs_lcc, crs_wgs84, always_xy=True)
    tf_nad83 = Transformer.from_crs(crs_lcc, crs_nad83, always_xy=True)

    # 4. Transform the 4 corner points and gather min/max
    corners = [(x_min, y_min),
               (x_min, y_max),
               (x_max, y_min),
               (x_max, y_max)]

    # WGS84
    lons_w, lats_w = zip(*(tf_wgs84.transform(x,y) for x,y in corners))
    print(f"WGS84 (EPSG:4326) lon range: {min(lons_w):.4f} → {max(lons_w):.4f}°")
    print(f"WGS84 (EPSG:4326) lat range: {min(lats_w):.4f} → {max(lats_w):.4f}°\n")

    # NAD83
    lons_n, lats_n = zip(*(tf_nad83.transform(x,y) for x,y in corners))
    print(f"NAD83 (EPSG:4269) lon range: {min(lons_n):.4f} → {max(lons_n):.4f}°")
    print(f"NAD83 (EPSG:4269) lat range: {min(lats_n):.4f} → {max(lats_n):.4f}°")


def convert_nc_to_h5(nc_path: str,
                     h5_path: str,
                     compression: str = "gzip",
                     compression_opts: int = 4):
    """
    Convert a NetCDF file to HDF5, transforming its native x/y grid into
    EPSG:4326 (WGS84 lon/lat), and preserving time & data_vars.

    - coords/:
        • lon (2D array, degrees_east)
        • lat (2D array, degrees_north)
        • time (1D int64 seconds since 1970-01-01T00:00:00)
    - data_vars/: all original variables (flipped along y if needed)
    """
    # 1. Open source dataset
    ds = xr.open_dataset(nc_path)

    # 2. Read & (optionally) flip the y‐coordinate so north→south
    x = ds.x.values
    y = ds.y.values[::-1]
    # convert to metres if units="km"
    unit = ds.x.attrs.get("units", "").lower()
    factor = 1000.0 if unit == "km" else 1.0
    X, Y = np.meshgrid(x * factor, y * factor)

    # 3. Extract the file’s projection (CF grid_mapping) → CRS
    gm = next((n for n,v in ds.variables.items()
               if "grid_mapping_name" in v.attrs), None)
    if gm is None:
        raise ValueError("No CF grid_mapping found in dataset.")
    src_crs = CRS.from_cf(ds[gm].attrs)

    # 4. Build transformer to WGS84
    dst_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # 5. Transform the meshgrid to lon/lat
    lon2d, lat2d = transformer.transform(X, Y)

    # 6. Prepare output file
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    with h5py.File(h5_path, "w") as hf:
        # ─── coords ────────────────────────
        cgrp = hf.create_group("coords")
        cgrp.create_dataset("lon", data=lon2d,
                            compression=compression,
                            compression_opts=compression_opts)
        cgrp.create_dataset("lat", data=lat2d,
                            compression=compression,
                            compression_opts=compression_opts)

        # time: convert datetime64 → int64 seconds since epoch
        t = ds.time.values.astype("datetime64[s]").astype("int64")
        ts = cgrp.create_dataset("time", data=t,
                                 compression=compression,
                                 compression_opts=compression_opts)
        ts.attrs.update(ds.time.attrs)
        ts.attrs.setdefault("units", "seconds since 1970-01-01T00:00:00")

        # ─── data_vars ────────────────────
        dgrp = hf.create_group("data_vars")
        for name, da in ds.data_vars.items():
            arr = da.values
            # flip along the original 'y' dim if present
            if "y" in da.dims:
                axis = da.dims.index("y")
                arr = np.flip(arr, axis=axis)
            ds_kwargs = {}
            if arr.ndim > 0 and compression:
                ds_kwargs = {"compression": compression,
                             "compression_opts": compression_opts}
            dset = dgrp.create_dataset(name, data=arr, **ds_kwargs)
            dset.attrs.update(da.attrs)


def clip_nc_to_gifs(
    shapefile_path: str,
    nc_file_path: str,
    var_name: str,
    output_dir: str
):
    """
    1. Load watershed and reproject to LCC.
    2. Open .nc, set its spatial dims & assign the same LCC CRS.
    3. Clip the DataArray in LCC.
    4. Loop over time and save each day's field as a GIF.
    """

    # ─── 1. Watershed → LCC ───
    lcc_proj = (
        "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 "
        "+lon_0=-100 +x_0=0 +y_0=0 +units=km +ellps=WGS84 +no_defs"
    )
    print(f"Loading watershed from {shapefile_path}...")
    ws = gpd.read_file(shapefile_path)
    ws_lcc = ws.to_crs(lcc_proj)

    # ─── 2. Open NetCDF & grab DataArray ───
    ds = xr.open_dataset(nc_file_path)
    da = ds[var_name]

    # ─── 2a. Tell rioxarray which dims are X/Y ───
    # (your dims are ('time','y','x'), so:)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    # ─── 2b. Assign (not reproject) the LCC CRS ───
    da = da.rio.write_crs(lcc_proj, inplace=False)

    # ─── 3. Clip in LCC ───
    da_clip = da.rio.clip(ws_lcc.geometry, ws_lcc.crs, drop=True)
    # (Now da_clip.x and da_clip.y will be roughly within your watershed bounds.)

    # ─── 4. Save each day as a GIF ───
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for t in da_clip.time.values:
        single = da_clip.sel(time=t)
        date_str = np.datetime_as_string(t, unit="D")
        out_path = Path(output_dir) / f"{date_str}.gif"

        # 1) Render to an in-memory PNG
        buf = io.BytesIO()
        fig, ax = plt.subplots(figsize=(8, 6))
        single.plot(ax=ax, cmap="viridis")
        ax.set_title(f"{var_name} — {date_str}")
        ax.set_axis_off()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        # 2) Convert PNG buffer to GIF via Pillow
        img = Image.open(buf)
        # Convert to "P" mode for GIF palette
        img = img.convert("P", palette=Image.ADAPTIVE)
        img.save(out_path, format="GIF")

        print(f"Saved {out_path}")
        

def clip_nc_to_h5(
    shapefile_path: str,
    nc_file_path: str,
    var_name: str,
    output_h5_path: str,
    compression: str = "gzip",
    compression_opts: int = 4
):
    """
    1. Load watershed and reproject to LCC.
    2. Open .nc, set its spatial dims & assign the same LCC CRS.
    3. Clip the DataArray in LCC.
    4. Flip it vertically (invert y).
    5. Save each day's 2D array as its own dataset in an .h5 file.
    """

    # ─── 1. Watershed → LCC ───
    lcc_proj = (
        "+proj=lcc +lat_1=25 +lat_2=60 +lat_0=42.5 "
        "+lon_0=-100 +x_0=0 +y_0=0 +units=km +ellps=WGS84 +no_defs"
    )
    ws = gpd.read_file(shapefile_path)
    ws_lcc = ws.to_crs(lcc_proj)

    # ─── 2. Open NetCDF & grab DataArray ───
    ds = xr.open_dataset(nc_file_path)
    da = ds[var_name]

    # ─── 2a. Tell rioxarray which dims are X/Y ───
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    # ─── 2b. Assign (not reproject) the LCC CRS ───
    da = da.rio.write_crs(lcc_proj, inplace=False)

    # ─── 3. Clip in LCC ───
    da_clip = da.rio.clip(ws_lcc.geometry, ws_lcc.crs, drop=True)
    # dims: ('time','y','x')

    # ─── 4. Flip vertically ───
    da_flip = da_clip.isel(y=slice(None, None, -1))

    # ─── 5. Write to HDF5 ───
    Path(output_h5_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_h5_path, "w") as hf:
        # x coords (unchanged)
        hf.create_dataset("x", data=da_flip.x.values)
        hf["x"].attrs["units"] = "km"

        # y coords (now reversed)
        hf.create_dataset("y", data=da_flip.y.values)
        hf["y"].attrs["units"] = "km"

        # time as ISO strings
        times = np.array([
            np.datetime_as_string(t, unit="D")
            for t in da_flip.time.values
        ], dtype="S10")
        hf.create_dataset("time", data=times)
        hf["time"].attrs["format"] = "YYYY-MM-DD"

        # group for daily data
        data_grp = hf.create_group("data")
        for t_idx, t in enumerate(da_flip.time.values):
            date_str = np.datetime_as_string(t, unit="D")
            ds_daily = data_grp.create_dataset(
                name=date_str,
                data=da_flip.isel(time=t_idx).values,
                compression=compression,
                compression_opts=compression_opts
            )
            # copy over attrs from the DataArray
            for k, v in da_flip.attrs.items():
                ds_daily.attrs[k] = v

    print(f"All days written (and vertically flipped) to {output_h5_path}")
    
    
            
if __name__ == "__main__":
    # — User config — 
    nc_path = "/data/HydroTransformer/pcp/prcp_2000subset6.nc"
    h5_path = "/data/HydroTransformer/pcp/prcp_2000subset6_raw.h5"
    # # ————————
    # print_extent_and_region(nc_path)
    
    # exit()
    
    # clip_nc_to_gifs(
    #     "data/processed/streamflow_data/watershed_shapefiles/4116000/4116000_watershed.shp",
    #     "/data/HydroTransformer/pcp/prcp_2000subset6.nc",
    #     var_name="prcp",
    #     output_dir="output/gifs"
    # )
    
    clip_nc_to_gifs(
        "../data/processed/streamflow_data/watershed_shapefiles/4116000/4116000_watershed.shp",
        "/data/HydroTransformer/pcp/daymet_v4_daily_na_dayl_2018.nc",
        var_name="dayl",
        output_dir="output/gifs"
    )
    
    # clip_nc_to_h5(
    #     "data/raw/Michigan/Final_Michigan_Map/Watershed_Boundary_Intersect_Michigan.shp",
    #     "/data/HydroTransformer/pcp/prcp_2000subset6.nc",
    #     var_name="prcp",
    #     output_h5_path="/data/HydroTransformer/pcp/prcp_2000subset6_MI.h5",
    # )
    # convert_nc_to_h5(nc_path, h5_path)