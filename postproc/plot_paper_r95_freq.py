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
out_path = os.path.join(fig_dir, "r95_nbEvents_freq_spatial_map_and_error_Annual.png")

models = ["GLM", "CNN", "Unet", "Vit"]
name_display = {"MSWEP": "MSWEP", "GLM": "GLM", "CNN": "CNN", "Unet": "U-Net", "Vit": "ViT"}

# Load shapefile
shapefile = utils.get_shapefile(region)
morocco_gdf = gpd.read_file(shapefile).dissolve()
morocco = morocco_gdf.boundary
bounds = morocco_gdf.total_bounds

# --- Load Data ---
# For extreme indices, the values are already computed in the metric files
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

# --- Plotting ---
fig = plt.figure(figsize=(24, 15), dpi=300)
# We want to center a 4-panel row under a 5-panel row
gs = gridspec.GridSpec(2, 20, figure=fig, hspace=0.45, wspace=0.0)

# --- Row 1: R95 Frequency ---
freq_levels = [0, 2, 4, 6, 8, 10, 15]
cmap_freq = mcolors.LinearSegmentedColormap.from_list("custom_freq", ["white", "lightyellow", "gold", "darkorange", "crimson", "purple", "darkmagenta"], N=256)
norm_freq = mcolors.BoundaryNorm(freq_levels, ncolors=cmap_freq.N, extend='max')

axes_freq = []
# 5 panels, each span 4 cols (20 cols total)
col_spans = [(0,4), (4,8), (8,12), (12,16), (16,20)]

for i, (name, data) in enumerate(freq_dict.items()):
    print(f"Plotting frequency for {name}...")
    ax = fig.add_subplot(gs[0, col_spans[i][0]:col_spans[i][1]], projection=ccrs.PlateCarree())
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
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    vals = data.values
    min_val, mean_val, max_val = float(np.nanmin(vals)), float(np.nanmean(vals)), float(np.nanmax(vals))
    
    display_name = name_display.get(name, name)
    ax.set_title(f"{display_name}\n", fontsize=15, fontweight='bold', pad=15)
    ax.text(0.5, 1.03, f"Min: {min_val:.1f}  |  Mean: {mean_val:.1f}  |  Max: {max_val:.1f}", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='#333333')
            
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

fig.text(0.12, 0.88, '(a)', fontsize=20, fontweight='bold', va='top', ha='right')

# Colorbar for Frequency
cbar_ax1 = fig.add_axes([0.3, 0.49, 0.4, 0.02])
cbar1 = fig.colorbar(im_freq, cax=cbar_ax1, orientation='horizontal', ticks=freq_levels, extend='max')
cbar1.set_label("R95 Frequency (Days/Year)", fontsize=14, fontweight='bold')
cbar1.ax.tick_params(labelsize=11)

# --- Row 2: Frequency Error ---
bias_levels = [-10, -8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8, 10]
BIAS_RDBU_WHITE = ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#ffffff", "#ffffff", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]
# Use exactly the same logic as utils.py to retain identical colormap
cmap_bias = mcolors.LinearSegmentedColormap.from_list("custom_bias", BIAS_RDBU_WHITE, N=len(bias_levels) + 1)
norm_bias = mcolors.BoundaryNorm(bias_levels, ncolors=cmap_bias.N, extend='both')

axes_error = []
# 4 panels, each span 5 cols (20 cols total)
col_spans_error = [(0,5), (5,10), (10,15), (15,20)]

for i, (name, data) in enumerate(error_dict.items()):
    print(f"Plotting error for {name}...")
    ax = fig.add_subplot(gs[1, col_spans_error[i][0]:col_spans_error[i][1]], projection=ccrs.PlateCarree())
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
    ax.set_title(f"{display_name} - MSWEP\n", fontsize=15, fontweight='bold', pad=15)
    ax.text(0.5, 1.03, f"Min: {min_val:.1f}  |  Mean: {mean_val:.1f}  |  Max: {max_val:.1f}", 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=12, color='#333333')
            
    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

fig.text(0.12, 0.44, '(b)', fontsize=20, fontweight='bold', va='top', ha='right')

# Colorbar for Bias
cbar_ax2 = fig.add_axes([0.3, 0.05, 0.4, 0.02])
cbar2 = fig.colorbar(im_error, cax=cbar_ax2, orientation='horizontal', ticks=bias_levels, extend='both')
cbar2.set_label("R95 Frequency Error (Days/Year)", fontsize=14, fontweight='bold')
cbar2.ax.tick_params(labelsize=11)

os.makedirs(fig_dir, exist_ok=True)
fig.savefig(out_path, bbox_inches='tight', dpi=300)
print(f"Publication-ready figure saved to {out_path}")
