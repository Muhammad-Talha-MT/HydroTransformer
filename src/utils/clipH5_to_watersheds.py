#!/usr/bin/env python3
import os
import glob
import h5py
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize

# ───────── CONFIGURATION ─────────
h5_path         = "/data/PRISM/PRISM.h5"
shapefile_dir   = "/home/talhamuh/water-research/HydroTransformer/data/processed/streamflow_data/watershed_shapefiles"
output_h5_file  = "/data/HydroTransformer/watershed_timeseries.h5"
params          = ["ppt", "tmin", "tmax"]
years           = list(range(1990, 2023))
# ───────────────────────────────────

def main():
    # 1) Open output HDF5
    with h5py.File(output_h5_file, "w") as h5_out:
        # 2) Read PRISM grid coords & infer H, W
        with h5py.File(h5_path, "r") as f:
            row      = f["coords/row"][:]
            col      = f["coords/col"][:]
            lat_flat = f["coords/lat"][:]
            lon_flat = f["coords/lon"][:]
            # sample to get H, W
            _, H, W = f["ppt"]["1990"][:].shape

        # 3) Build full 2D lat/lon grids & affine transform
        lat_grid = np.full((H, W), np.nan)
        lon_grid = np.full((H, W), np.nan)
        lat_grid[row, col] = lat_flat
        lon_grid[row, col] = lon_flat

        minx, maxx = np.nanmin(lon_grid), np.nanmax(lon_grid)
        miny, maxy = np.nanmin(lat_grid), np.nanmax(lat_grid)
        transform  = from_bounds(minx, miny, maxx, maxy, W, H)

        # 4) Loop through each watershed shapefile
        for shp_path in glob.glob(os.path.join(shapefile_dir, "**/*.shp"), recursive=True):
            ws_name = os.path.splitext(os.path.basename(shp_path))[0]
            print(f"Processing {ws_name}...")

            # read & reproject to lat/lon
            gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
            # union_all replaces unary_union
            geom = [gdf.geometry.union_all().__geo_interface__]

            # create group & store coords
            ws_grp = h5_out.create_group(ws_name)
            
            
            cord   = ws_grp.create_group("CORD")
            cord.create_dataset("row", data=row)
            cord.create_dataset("col", data=col)
            cord.create_dataset("lat", data=lat_flat)
            cord.create_dataset("lon", data=lon_flat)

            # rasterize entire grid once
            mask_full = rasterize(
                [(geom[0], 1)],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype="uint8"
            ).astype(bool)

            # compute static bounding box on the full mask
            rows_any = mask_full.any(axis=1)
            cols_any = mask_full.any(axis=0)
            r0, r1 = np.where(rows_any)[0][[0, -1]]
            c0, c1 = np.where(cols_any)[0][[0, -1]]
            mask_win = mask_full[r0:r1+1, c0:c1+1]

            # 5) For each parameter and year, read the small window once & vectorize mask
            with h5py.File(h5_path, "r") as f_in:
                for param in params:
                    print(f"  • {param}")
                    param_grp = ws_grp.create_group(param)

                    for year in years:
                        ds_year = f_in[param][str(year)]
                        # read all days × window in one hyperslab
                        data_win = ds_year[:, r0:r1+1, c0:c1+1]  # shape: (days, h, w)
                        # apply mask across time dimension in one vectorized call
                        masked = np.where(mask_win[None, :, :], data_win, np.nan)
                        # save out with compression & chunking
                        param_grp.create_dataset(
                            str(year),
                            data=masked,
                            compression="gzip",
                            chunks=True
                        )

            print(f"✅ Data for watershed '{ws_name}' saved.")

        print(f"✅ All watershed data written to {output_h5_file}")

if __name__ == "__main__":
    main()
