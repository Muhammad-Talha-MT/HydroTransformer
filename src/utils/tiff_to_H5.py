import os
import rioxarray
import h5py
import numpy as np

def convert_tiff_to_hdf(tiff_folders, hdf_file):
    # Ensure the parent directory of the HDF5 file exists
    os.makedirs(os.path.dirname(hdf_file), exist_ok=True)
    
    # Create or open the HDF5 file
    with h5py.File(hdf_file, 'a') as hdf5_file:
        # Loop through each folder (each folder corresponds to a variable)
        for folder_name in tiff_folders:
            variable_name = os.path.basename(folder_name)
            folder_files = [f for f in os.listdir(folder_name) if f.endswith('.tif')]

            for tiff_file in folder_files:
                tiff_path = os.path.join(folder_name, tiff_file)
                
                # Read the TIFF file using rioxarray
                raster = rioxarray.open_rasterio(tiff_path)

                # Replace value 32767 with 0 in DEM_clips (if the variable is DEM)
                if 'DEM' in variable_name:
                    raster_values = np.where(raster.values == 32767, 0, raster.values)
                else:
                    raster_values = raster.values

                # Invert the data along the X-axis (horizontal flip)
                inverted_data = np.fliplr(raster_values)

                # Extract watershed name from the filename (before '_awc.tif')
                watershed_name = tiff_file.split('_')[0]  # Adjust based on your naming convention
                # Append '_watershed' to the watershed name
                watershed_name = f"{watershed_name}_watershed"
                # Create a group for the watershed if it doesn't exist
                if watershed_name not in hdf5_file:
                    watershed_group = hdf5_file.create_group(watershed_name)
                else:
                    watershed_group = hdf5_file[watershed_name]

                # Store the inverted raster data under the appropriate group and variable
                # Use the variable (folder) name and TIFF file base name as dataset name
                dataset_name = os.path.splitext(tiff_file)[0]
                variable_group = watershed_group.create_group(variable_name) if variable_name not in watershed_group else watershed_group[variable_name]
                variable_group.create_dataset(dataset_name, data=inverted_data, compression="gzip")

                print(f"Converted {tiff_file} to HDF5 at {watershed_name}/{variable_name}/{dataset_name}")

# Example usage:
# List of folders corresponding to different variables
tiff_folders = [
    '../../data/processed/static_parameters_data/awc_clips',   # Folder for AWC (Available Water Capacity) data
    '../../data/processed/static_parameters_data/DEM_clips',   # Folder for DEM (Digital Elevation Model) data
    '../../data/processed/static_parameters_data/fc_clips',    # Folder for Forest Cover data
    '../../data/processed/static_parameters_data/NLCD_clips',  # Folder for NLCD (National Land Cover Database) data
    '../../data/processed/static_parameters_data/soil_clips'   # Folder for Soil data
]

hdf_file = '../../data/processed/static_parameters_data/file.h5'  # Path to the output HDF5 file

convert_tiff_to_hdf(tiff_folders, hdf_file)
