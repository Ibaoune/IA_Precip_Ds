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
metric_dir = os.path.join(root_path, "results", "GLM_CNN_Unet_Vit_retained", "2006-01-01_2020-12-31", "allmorr", "test", "cdd", "cdd_0.5", "results")
fig_dir = os.path.join(root_path, "results", "GLM_CNN_Unet_Vit_retained", "2006-01-01_2020-12-31", "allmorr", "test", "cdd", "cdd_0.5", "figures", "all")
os.makedirs(fig_dir, exist_ok=True)

freq_out_path = os.path.join(fig_dir, "spatial_cdd_comparison_Annual.png")
error_out_path = os.path.join(fig_dir, "spatial_cdd_error_Annual.png")

models = ["GLM", "CNN", "Unet", "Vit"]
name_display = {"MSWEP": "MSWEP", "GLM": "GLM", "CNN": "CNN", "Unet": "U-Net", "Vit": "ViT"}

# Load shapefile
shapefile = utils.get_shapefile(region)
morocco_gdf = gpd.read_file(shapefile).dissolve()
morocco = morocco_gdf.boundary
bounds = morocco_gdf.total_bounds

# --- Load Data ---
model_paths = {
    m: os.path.join(metric_dir, f"{m}_pr_allmorr_calcul_land_strategy_per_year_corr_per_year_{period}.nc")
    for m in models
}
ref_path = os.path.join(metric_dir, f"mswep_pr_allmorr_calcul_land_strategy_per_year_corr_per_year_{period}.nc")

# Read MSWEP
ds_ref = xr.open_dataset(ref_path)
ref_period_mean = ds_ref.mean("year") if "year" in ds_ref.dims else ds_ref
ref_period_mean = ref_period_mean["cdd"]

freq_dict = {"MSWEP": ref_period_mean}
error_dict = {}

for name, path in model_paths.items():
    ds = xr.open_dataset(path)
    mean_ds = ds.mean("year") if "year" in ds.dims else ds
    mean_ds = mean_ds["cdd"]
    freq_dict[name] = mean_ds
    error_dict[name] = mean_ds - ref_period_mean

# ==========================================
# 1. Plot Individual CDD Comparison
# ==========================================
fig_freq = plt.figure(figsize=(24, 7), dpi=300)
gs_freq = gridspec.GridSpec(1, 5, figure=fig_freq, wspace=0.0)

freq_levels = [0, 10, 20, 30, 40, 60, 80, 100, 150, 200, 250, 300, 400]
base_cmap_freq = plt.get_cmap("YlOrBr", 13)
colors_freq = [base_cmap_freq(i) for i in range(13)]
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
cbar1.set_label("CDD_0.5 (days)", fontsize=14, fontweight='bold')
cbar1.ax.tick_params(labelsize=11)

fig_freq.savefig(freq_out_path, bbox_inches='tight', dpi=300)
print(f"Saved {freq_out_path}")

# ==========================================
# 2. Plot Individual CDD Error
# ==========================================
fig_bias = plt.figure(figsize=(24, 7), dpi=300)
gs_bias = gridspec.GridSpec(1, 4, figure=fig_bias, wspace=0.0)

bias_levels = [-100, -80, -60, -40, -20, -10, 0, 10, 20, 40, 60, 80, 100]
base_cmap_bias = plt.get_cmap("RdBu", 14)
colors_bias = [base_cmap_bias(i) for i in range(14)]
colors_bias[6] = (1.0, 1.0, 1.0, 1.0) # [-10, 0] is white
colors_bias[7] = (1.0, 1.0, 1.0, 1.0) # [0, 10] is white
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
cbar2.set_label("CDD Error (days)", fontsize=14, fontweight='bold')
cbar2.ax.tick_params(labelsize=11)

fig_bias.savefig(error_out_path, bbox_inches='tight', dpi=300)
print(f"Saved {error_out_path}")
