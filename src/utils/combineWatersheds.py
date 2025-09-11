import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR    = 'data/processed/streamflow_data/watershed_shapefiles'
OUTPUT_FILE = 'data/processed/streamflow_data/watershed_MI_61/watersheds_combined.shp'

# -----------------------------------------------------------------------------
# MAKE SURE OUTPUT DIRECTORY EXISTS
# -----------------------------------------------------------------------------
out_dir = os.path.dirname(OUTPUT_FILE)
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# COLLECT AND UNION POLYGONS FOR EACH WATERSHED
# -----------------------------------------------------------------------------
records = []

for folder_name in sorted(os.listdir(BASE_DIR)):
    folder_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    shp_paths = glob.glob(os.path.join(folder_path, '*.shp'))
    if not shp_paths:
        print(f"⚠️  No .shp files in {folder_path}, skipping.")
        continue

    # Read & concatenate all shapefiles in this folder
    gdfs = [gpd.read_file(shp) for shp in shp_paths]
    # Reproject all to the CRS of the first file
    target_crs = gdfs[0].crs
    merged = gpd.GeoDataFrame(
        pd.concat([g.to_crs(target_crs) for g in gdfs], ignore_index=True),
        crs=target_crs
    )

    # Union into one polygon per watershed
    union_geom = unary_union(merged.geometry)
    records.append({
        'watershed_id': folder_name[:10],  # keep ≤10 chars so DBF field names don’t get cut
        'geometry': union_geom
    })

# -----------------------------------------------------------------------------
# BUILD FINAL GeoDataFrame AND WRITE OUT SHAPEFILE
# -----------------------------------------------------------------------------
final_gdf = gpd.GeoDataFrame(records, crs=target_crs)

# Write ESRI Shapefile
final_gdf.to_file(OUTPUT_FILE, driver='ESRI Shapefile')

print(f"✅  Wrote {len(final_gdf)} watersheds to Shapefile at {OUTPUT_FILE}")
