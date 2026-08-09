import os
import sys
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.gridspec as gridspec

# Add postproc/src to path for utils
sys.path.append("/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/src")
import utils

# --- Configuration ---
period = "Annual"
region = "allmorr"
root_path = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc"
metric_dir = os.path.join(root_path, "results", "GLM_CNN_Unet_Vit_retained", "2006-01-01_2020-12-31", "allmorr", "test", "r95_nbEvents_freq", "results")
fig_dir = os.path.join(root_path, "results", "GLM_CNN_Unet_Vit_retained", "2006-01-01_2020-12-31", "allmorr", "test", "r95_nbEvents_freq", "figures", "all")
os.makedirs(fig_dir, exist_ok=True)

freq_out_path = os.path.join(fig_dir, "r95_nbEvents_freq_spatial_map_Annual.png")
error_out_path = os.path.join(fig_dir, "spatial_frequency_error_Annual.png")

models = ["GLM", "CNN", "Unet", "Vit"]
name_display = {"MSWEP": "MSWEP", "GLM": "GLM", "CNN": "CNN", "Unet": "U-Net", "Vit": "ViT"}

# Load shapefile
shapefile = utils.get_shapefile(region)
morocco_gdf = gpd.read_file(shapefile).dissolve()
morocco = morocco_gdf.boundary
bounds = morocco_gdf.total_bounds

# --- Load Data ---
model_paths = {
    m: os.path.join(metric_dir, f"{m}_pr_allmorr_calcul_land_strategy_daily_first_corr_per_year_{period}.nc")
    for m in models
}
ref_path = os.path.join(metric_dir, f"mswep_pr_allmorr_calcul_land_strategy_daily_first_corr_per_year_{period}.nc")

# Read MSWEP
ds_ref = xr.open_dataset(ref_path)
ref_period_mean = ds_ref.mean("year") if "year" in ds_ref.dims else ds_ref
ref_period_mean = ref_period_mean["r95p"]

freq_dict = {"MSWEP": ref_period_mean}
error_dict = {}

for name, path in model_paths.items():
    ds = xr.open_dataset(path)
    mean_ds = ds.mean("year") if "year" in ds.dims else ds
    mean_ds = mean_ds["r95p"]
    freq_dict[name] = mean_ds
    error_dict[name] = mean_ds - ref_period_mean

# ==========================================
# 1. Plot Individual R95 Frequency
# ==========================================
fig_freq = plt.figure(figsize=(24, 7), dpi=300)
gs_freq = gridspec.GridSpec(1, 5, figure=fig_freq, wspace=0.0)

freq_levels = [0, 0.5, 1, 2, 4, 6, 8, 10, 15]
# 8 bins. We need 9 colors total (including max extension)
base_cmap_freq = mcolors.LinearSegmentedColormap.from_list(
    "custom_freq", ["white", "lightyellow", "gold", "darkorange", "crimson", "purple", "darkmagenta"], N=256)
colors_freq = [base_cmap_freq(i / 8) for i in range(9)]
colors_freq[0] = (1.0, 1.0, 1.0, 1.0) # [0, 0.5] is white
cmap_freq = mcolors.ListedColormap(colors_freq)
norm_freq = mcolors.BoundaryNorm(freq_levels, ncolors=cmap_freq.N, extend='max')

axes_freq = []
for i, (name, data) in enumerate(freq_dict.items()):
    ax = fig_freq.add_subplot(gs_freq[0, i], projection=ccrs.PlateCarree())
    axes_freq.append(ax)
    
    extent = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
    im_freq = ax.imshow(data, extent=extent, cmap=cmap_freq, norm=norm_freq, origin="upper", transform=ccrs.PlateCarree())
    
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
    
    vals = data.values
    min_val, mean_val, max_val = float(np.nanmin(vals)), float(np.nanmean(vals)), float(np.nanmax(vals))
    
    display_name = name_display.get(name, name)
    ax.set_title(f"{display_name}\n", fontsize=18, fontweight='bold', pad=15)
    ax.text(0.5, 1.03, f"Min: {min_val:.1f}  |  Mean: {mean_val:.1f}  |  Max: {max_val:.1f}", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=14, color='#333333')
    
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

cbar_ax1 = fig_freq.add_axes([0.3, -0.05, 0.4, 0.03])
cbar1 = fig_freq.colorbar(im_freq, cax=cbar_ax1, orientation='horizontal', ticks=freq_levels, extend='max')
cbar1.set_label("R95 Frequency (Days/Year)", fontsize=14, fontweight='bold')
cbar1.ax.tick_params(labelsize=11)

fig_freq.savefig(freq_out_path, bbox_inches='tight', dpi=300)
print(f"Saved {freq_out_path}")

# ==========================================
# 2. Plot Individual Spatial Frequency Error
# ==========================================
fig_bias = plt.figure(figsize=(24, 7), dpi=300)
gs_bias = gridspec.GridSpec(1, 4, figure=fig_bias, wspace=0.0)

bias_levels = [-10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10]
# 12 bins. We need 14 colors total (including min and max extensions)
base_cmap_bias = plt.get_cmap("RdBu", 14)
colors_bias = [base_cmap_bias(i) for i in range(14)]
colors_bias[6] = (1.0, 1.0, 1.0, 1.0) # [-1, 0] is white
colors_bias[7] = (1.0, 1.0, 1.0, 1.0) # [0, 1] is white
cmap_bias = mcolors.ListedColormap(colors_bias)
norm_bias = mcolors.BoundaryNorm(bias_levels, ncolors=cmap_bias.N, extend='both')

axes_error = []
for i, (name, data) in enumerate(error_dict.items()):
    ax = fig_bias.add_subplot(gs_bias[0, i], projection=ccrs.PlateCarree())
    axes_error.append(ax)
    
    extent = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
    im_error = ax.imshow(data, extent=extent, cmap=cmap_bias, norm=norm_bias, origin="upper", transform=ccrs.PlateCarree())
    
    ax.coastlines(resolution='10m', color='0.3', linewidth=0.8)
    morocco.plot(ax=ax, edgecolor="black", linewidth=1.2, transform=ccrs.PlateCarree())
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.3, color='gray')
    gl.top_labels = False
    gl.right_labels = False
    if i > 0:
        gl.left_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    vals = data.values
    min_val, mean_val, max_val = float(np.nanmin(vals)), float(np.nanmean(vals)), float(np.nanmax(vals))
    
    display_name = name_display.get(name, name)
    ax.set_title(f"{display_name} - MSWEP\n", fontsize=18, fontweight='bold', pad=15)
    ax.text(0.5, 1.03, f"Min: {min_val:.1f}  |  Mean: {mean_val:.1f}  |  Max: {max_val:.1f}", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=14, color='#333333')
            
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

cbar_ax2 = fig_bias.add_axes([0.3, -0.05, 0.4, 0.03])
cbar2 = fig_bias.colorbar(im_error, cax=cbar_ax2, orientation='horizontal', ticks=bias_levels, extend='both')
cbar2.set_label("R95 Frequency Error (Days/Year)", fontsize=14, fontweight='bold')
cbar2.ax.tick_params(labelsize=11)

fig_bias.savefig(error_out_path, bbox_inches='tight', dpi=300)
print(f"Saved {error_out_path}")
