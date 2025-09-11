import xarray as xr
import matplotlib.pyplot as plt
import imageio
import os

# 1. Configuration
nc_path      = "/data/HydroTransformer/pcp/prcp_1980subset6.nc"
var_name     = "prcp"                # adjust to the name of your variable
time_dim     = "time"                # often "time"
output_gif   = "precipitation.gif"
frames_dir   = "gif_frames"
fps          = 3                     # frames per second in the final GIF

# 2. Create a temporary directory for frame images
os.makedirs(frames_dir, exist_ok=True)

# 3. Load the dataset (lazy load)
ds = xr.open_dataset(nc_path)

# 4. Loop over each time slice, plot and save as PNG
png_files = []
for i, t in enumerate(ds[time_dim].values):
    fig, ax = plt.subplots(figsize=(6,4))
    # Plot the spatial field
    im = ds[var_name].isel({time_dim: i}).plot(
        ax=ax,
        cmap="viridis",
        add_colorbar=True,
        cbar_kwargs={"label": ds[var_name].units if "units" in ds[var_name].attrs else ""}
    )
    ax.set_title(f"{var_name} at {str(t)[:19]}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Save frame
    png_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    png_files.append(png_path)

# 5. Build the GIF
with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
    for filename in png_files:
        image = imageio.imread(filename)
        writer.append_data(image)

# 6. (Optional) Clean up the frames directory
# import shutil
# shutil.rmtree(frames_dir)

print(f"GIF saved to {output_gif}")