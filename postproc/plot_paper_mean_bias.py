import os
import sys
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Add postproc/src to path for utils
sys.path.append("/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/src")
import utils

# --- Configuration ---
period = "Annual"
region = "allmorr"
root_path = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc"
metric_dir = os.path.join(root_path, "results", "GLM_CNN_Unet_Vit_retained", "2006-01-01_2020-12-31", "allmorr", "test", "bias", "results")
out_path = os.path.join(root_path, "results", "GLM_CNN_Unet_Vit_retained", "2006-01-01_2020-12-31", "allmorr", "test", "bias", "figures", "all", "mean_precip_and_spatial_bias_error_Annual.png")

models = ["GLM", "CNN", "Unet", "Vit"]
name_display = {"MSWEP": "MSWEP", "GLM": "GLM", "CNN": "CNN", "Unet": "U-Net", "Vit": "ViT"}

# Load shapefile
shapefile = utils.get_shapefile(region)
morocco_gdf = gpd.read_file(shapefile).dissolve()
morocco = morocco_gdf.boundary
bounds = morocco_gdf.total_bounds

# --- Load Data ---
model_paths = {
    m: os.path.join(metric_dir, f"{m}_pr_allmorr_calcul_land_strategy_mean_first_corr_per_year_{period}.nc")
    for m in models
}

data_raw = utils.load_metric_results(model_paths, period=period)

ref_path = "/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc"
ds_ref = xr.open_dataset(ref_path)
ds_ref = utils.mask_dataset(ds_ref, region=region, only_land=True, only_morocco=True)
ds_ref = ds_ref.sel(time=slice('2006-01-01', '2020-12-31'))
ref_period_mean = ds_ref["precipitation"].mean("time")

mean_dict = {"MSWEP": ref_period_mean}
bias_dict = {}

for name, ds in data_raw.items():
    mean_ds = ds.mean("year") if "year" in ds.dims else ds
    bias_dict[name] = mean_ds
    mean_dict[name] = mean_ds + ref_period_mean

# --- Plotting Setup ---
fig = plt.figure(figsize=(24, 14), dpi=300)

import matplotlib.gridspec as gridspec
# Increased hspace to make clear room for the middle colorbar and avoid overlap with row (b) titles
gs = gridspec.GridSpec(2, 20, figure=fig, hspace=0.45, wspace=0.0)

# --- Row 1: Climatology (Mean) ---
precip_levels = [0, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]
n_bins_mean = len(precip_levels) - 1 # 12 bins inside boundaries
# Need 12 colors for inside + 1 for max extension = 13 colors total
base_cmap_mean = mcolors.LinearSegmentedColormap.from_list(
    "custom_precip", ["white", "lightblue", "darkblue", "darkgreen", "lightgreen", "yellow", "orange", "red"], N=256)
# Sample exactly 13 discrete colors from the continuous custom colormap
colors_mean = [base_cmap_mean(i / 12) for i in range(13)]
colors_mean[0] = (1.0, 1.0, 1.0, 1.0) # Ensure the first interval [0, 0.1] is pure white
cmap_mean = mcolors.ListedColormap(colors_mean)
norm_mean = mcolors.BoundaryNorm(precip_levels, ncolors=cmap_mean.N, extend='max')

axes_mean = []
for i, (name, data) in enumerate(mean_dict.items()):
    ax = fig.add_subplot(gs[0, i*4:(i+1)*4], projection=ccrs.PlateCarree())
    axes_mean.append(ax)
    
    extent = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
    im_mean = ax.imshow(data, extent=extent, cmap=cmap_mean, norm=norm_mean, origin="upper", transform=ccrs.PlateCarree())
    
    ax.coastlines(resolution='10m', color='0.3', linewidth=0.8)
    morocco.plot(ax=ax, edgecolor="black", linewidth=1.2, transform=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    if i > 0:
        gl.left_labels = False 
    gl.bottom_labels = False 
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    min_val, mean_val, max_val = float(data.min()), float(data.mean()), float(data.max())
    
    # Clean hierarchical title
    display_name = name_display.get(name, name)
    ax.set_title(f"{display_name}\n", fontsize=15, fontweight='bold', pad=15)
    ax.text(0.5, 1.03, f"Min: {min_val:.1f}  |  Mean: {mean_val:.1f}  |  Max: {max_val:.1f}", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='#333333')
    
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

# (a) label far left
fig.text(0.05, 0.88, "(a)", fontsize=22, fontweight='bold', va='top')

# Colorbar for Mean (placed right between rows, slightly higher to avoid row (b) titles)
cbar_ax1 = fig.add_axes([0.3, 0.51, 0.4, 0.015]) # x, y, width, height
cbar1 = fig.colorbar(im_mean, cax=cbar_ax1, orientation='horizontal', ticks=precip_levels, extend='max')
cbar1.set_label("Mean Precipitation (mm/day)", fontsize=14, fontweight='bold')
cbar1.ax.tick_params(labelsize=11)

# --- Row 2: Bias ---
bias_levels = [-2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 2]
n_bins_bias = len(bias_levels) - 1 # 11 bins inside boundaries
# Need 11 + 2 extensions (both ends) = 13 colors total
base_cmap_bias = plt.get_cmap("RdBu", n_bins_bias + 2) 
colors_bias = [base_cmap_bias(i) for i in range(n_bins_bias + 2)]
# Make intervals [-0.1, 0] and [0, 0.1] white (indices 5 and 6, since 0 is < -2)
colors_bias[5] = (1.0, 1.0, 1.0, 1.0)
colors_bias[6] = (1.0, 1.0, 1.0, 1.0)
cmap_bias = mcolors.ListedColormap(colors_bias)
norm_bias = mcolors.BoundaryNorm(bias_levels, ncolors=cmap_bias.N, extend='both')

axes_bias = []
for i, (name, data) in enumerate(bias_dict.items()):
    ax = fig.add_subplot(gs[1, i*5:(i+1)*5], projection=ccrs.PlateCarree())
    axes_bias.append(ax)
    
    extent = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
    im_bias = ax.imshow(data, extent=extent, cmap=cmap_bias, norm=norm_bias, origin="upper", transform=ccrs.PlateCarree())
    
    ax.coastlines(resolution='10m', color='0.3', linewidth=0.8)
    morocco.plot(ax=ax, edgecolor="black", linewidth=1.2, transform=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    if i > 0:
        gl.left_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    min_val, mean_val, max_val = float(data.min()), float(data.mean()), float(data.max())
    
    display_name = name_display.get(name, name)
    ax.set_title(f"{display_name} - MSWEP\n", fontsize=15, fontweight='bold', pad=15)
    ax.text(0.5, 1.03, f"Min: {min_val:.1f}  |  Mean: {mean_val:.1f}  |  Max: {max_val:.1f}", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='#333333')
            
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

# (b) label far left (aligned vertically with (a), shifted down due to hspace increase)
fig.text(0.05, 0.42, "(b)", fontsize=22, fontweight='bold', va='top')

# Colorbar for Bias
cbar_ax2 = fig.add_axes([0.3, 0.05, 0.4, 0.015])
cbar2 = fig.colorbar(im_bias, cax=cbar_ax2, orientation='horizontal', ticks=bias_levels, extend='both')
cbar2.set_label("Spatial Bias Error (mm/day)", fontsize=14, fontweight='bold')
cbar2.ax.tick_params(labelsize=11)

# Make sure we don't use tight_layout() which could override manual positions
# Instead just leave it as manual layout since gridspec handles most spacing
plt.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Publication-ready figure saved to {out_path}")
