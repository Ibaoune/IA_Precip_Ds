"""
Author: M. El Aabaribaoune (@um6p)
Description: Generates custom, publication-ready figures for the research paper.
"""

#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import string

# Ensure high quality standard
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

from utils import (
    get_shapefile, BIAS_RDBU_WHITE
)

# --- Configuration ---
PRED_ROOT = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/inference/results/1979_2014"
DATASETS_DIR = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/data/datasets/datasets"
MSWEP_PATH = "/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc"
LMDZ35_NATIVE_PATH = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/data/lmdz/r35/hist/present/all_Mor/precip-hist.nc"
LMDZ250_NATIVE_PATH = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/data/lmdz/r250/amip/present/all_Mor/precip-amip.nc"

OUT_DIR = "results/inference_retained/figures/lmdz_paper"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ORDER = ["MSWEP", "LMDZ250", "LMDZ35", "U-Net", "CNN", "ViT"]
MODEL_COLORS = {
    'MSWEP': '#000000',
    'LMDZ250': '#F4A261',
    'LMDZ35': '#E76F51',
    'U-Net': '#1D3557',
    'CNN': '#457B9D',
    'ViT': '#E63946'
}

REGIONS = ["north", "north_east", "east", "south"]
SEASONS = {'DJF': [12, 1, 2], 'MAM': [3, 4, 5], 'JJA': [6, 7, 8], 'SON': [9, 10, 11]}

# --- Custom Colormaps ---
cmap_precip = mcolors.LinearSegmentedColormap.from_list("custom_precip", ["white", "lightblue", "darkblue", "darkgreen", "lightgreen", "yellow", "orange", "red"], N=256)
norm_precip = mcolors.BoundaryNorm([0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10], ncolors=cmap_precip.N, extend='max')

cmap_bias = mcolors.LinearSegmentedColormap.from_list("custom_bias", BIAS_RDBU_WHITE, N=13)
norm_bias = mcolors.BoundaryNorm([-5, -3, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 3, 5], ncolors=cmap_bias.N, extend='both')

norm_inc = mcolors.BoundaryNorm([-2, -1.5, -1, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 1, 1.5, 2], ncolors=cmap_bias.N, extend='both')

# --- Helper Functions ---
def add_panel_label(ax, text):
    ax.text(-0.05, 1.05, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')

def setup_map_axis(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', edgecolor='lightgray')
    ax.set_extent([-18, 0, 21, 37], crs=ccrs.PlateCarree())
    
    # Overlay Morocco boundary
    shapefile = get_shapefile("allmorr")
    morocco = gpd.read_file(shapefile).to_crs("EPSG:4326").dissolve()
    ax.add_geometries(morocco.geometry, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=1)
    
    return ax

def get_region_mask(ds, region):
    shapefile = get_shapefile(region)
    morocco = gpd.read_file(shapefile).to_crs("EPSG:4326").dissolve()
    morocco["name"] = ["morocco"]
    morocco = morocco.reset_index(drop=True)
    mask = regionmask.from_geopandas(morocco, names="name", name="morocco").mask(ds)
    return ~mask.isnull()

def load_standardized_lmdz(path):
    print(f"Loading native LMDZ from {path}")
    ds = xr.open_dataset(path)
    time_normalized = ds["time_counter"].dt.floor("D")
    ds = ds.assign_coords(time_counter=time_normalized)
    ds = ds.rename({"time_counter": "time"})
    if "precip" in ds: ds = ds.rename({"precip": "precipitation"})
    ds["precipitation"] = ds["precipitation"] * 86400.0
    return ds[["precipitation"]].sel(time=slice("1979-01-01", "2014-12-31"))

def load_10km_data():
    print("Loading 10km datasets...")
    mswep = xr.open_dataset(MSWEP_PATH)
    if "precipitation" not in mswep and "pr" in mswep: mswep = mswep.rename({"pr": "precipitation"})
    mswep = mswep[["precipitation"]].sel(time=slice("1979-01-01", "2014-12-31"))
    
    lmdz250_10km = xr.open_dataset(os.path.join(DATASETS_DIR, "raw_lmdz250_10km.nc"))[["precipitation"]]
    lmdz35_10km = xr.open_dataset(os.path.join(DATASETS_DIR, "raw_lmdz35_10km.nc"))[["precipitation"]]
    unet_10km = xr.open_dataset(os.path.join(PRED_ROOT, "unet_exp35_1x1_hybrid/unet_exp35_lmdz_250_present_true.nc"))[["precipitation"]]
    cnn_10km = xr.open_dataset(os.path.join(PRED_ROOT, "cnn_exp5/cnn_lmdz_250_present_true.nc"))[["precipitation"]]
    vit_10km = xr.open_dataset(os.path.join(PRED_ROOT, "vit_precip_exp21_best_hybrid/vit_lmdz_250_present_true.nc"))[["precipitation"]]
    
    return {
        "MSWEP": mswep, "LMDZ250": lmdz250_10km, "LMDZ35": lmdz35_10km,
        "U-Net": unet_10km, "CNN": cnn_10km, "ViT": vit_10km
    }

def calc_cdd(da):
    """Calculate Max CDD (<1mm/day) per year efficiently."""
    dry_days = (da < 1.0).astype(int)
    def cdd_1d(arr):
        if np.all(np.isnan(arr)): return np.nan
        c = 0; max_c = 0
        for x in arr:
            if x == 1:
                c += 1; max_c = max(max_c, c)
            else:
                c = 0
        return max_c
    
    cdd_yearly = xr.apply_ufunc(
        cdd_1d,
        dry_days.groupby('time.year'),
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask="allowed"
    )
    return cdd_yearly.mean(dim='year')

def save_fig(fig, name):
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    print(f"Saved {name}")
    plt.close(fig)

# --- Figure 1: Seasonal Precipitation Maps ---
def plot_figure_1(data):
    print("Generating Main Figure 1 (Seasonal Precipitation Maps)...")
    fig = plt.figure(figsize=(22, 15))
    labels = list(string.ascii_lowercase)
    
    mswep = data["MSWEP"]["precipitation"]
    mask_all = get_region_mask(mswep.mean(dim='time'), "allmorr")
    
    # Pre-compute season means for performance
    season_data = {}
    for model in MODEL_ORDER:
        ds = data[model]["precipitation"]
        # Fast groupby season
        season_mean = ds.groupby('time.season').mean(dim='time')
        season_data[model] = season_mean

    # xarray season codes: DJF, MAM, JJA, SON
    ordered_seasons = ['DJF', 'MAM', 'JJA', 'SON']
    plot_idx = 1
    
    for i, season in enumerate(ordered_seasons):
        for j, model in enumerate(MODEL_ORDER):
            ax = fig.add_subplot(4, 6, plot_idx, projection=ccrs.PlateCarree())
            setup_map_axis(ax)
            ds_season = season_data[model].sel(season=season).where(mask_all)
            
            im = ax.pcolormesh(ds_season.lon, ds_season.lat, ds_season, transform=ccrs.PlateCarree(), cmap=cmap_precip, norm=norm_precip)
            
            if i == 0: ax.set_title(model, loc='center', fontweight='bold', fontsize=12)
            if j == 0: ax.text(-0.15, 0.5, f'{season}', va='center', ha='center', rotation=90, transform=ax.transAxes, fontweight='bold', fontsize=14)
            add_panel_label(ax, f"({labels[plot_idx-1]})")
            plot_idx += 1
            
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
    fig.colorbar(im, cax=cbar_ax, extend='max').set_label('Mean precipitation (mm day⁻¹)', fontsize=12)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    save_fig(fig, "Fig_LMDZ_Seasonal_Precipitation_Maps")

# --- Figure 2: Seasonal Bias Maps ---
def plot_figure_2(data):
    print("Generating Main Figure 2 (Seasonal Bias Maps)...")
    fig = plt.figure(figsize=(18, 15))
    labels = list(string.ascii_lowercase)
    
    mswep = data["MSWEP"]["precipitation"]
    mask_all = get_region_mask(mswep.mean(dim='time'), "allmorr")
    
    season_data = {}
    for model in MODEL_ORDER:
        ds = data[model]["precipitation"]
        season_data[model] = ds.groupby('time.season').mean(dim='time')

    ordered_seasons = ['DJF', 'MAM', 'JJA', 'SON']
    plot_models = [m for m in MODEL_ORDER if m != "MSWEP"]
    plot_idx = 1
    
    for i, season in enumerate(ordered_seasons):
        mswep_season = season_data["MSWEP"].sel(season=season).where(mask_all)
        for j, model in enumerate(plot_models):
            ax = fig.add_subplot(4, 5, plot_idx, projection=ccrs.PlateCarree())
            setup_map_axis(ax)
            ds_season = season_data[model].sel(season=season).where(mask_all)
            bias = ds_season - mswep_season
            
            im = ax.pcolormesh(bias.lon, bias.lat, bias, transform=ccrs.PlateCarree(), cmap=cmap_bias, norm=norm_bias)
            
            if i == 0: ax.set_title(f"{model} - MSWEP", loc='center', fontweight='bold', fontsize=12)
            if j == 0: ax.text(-0.15, 0.5, f'{season}', va='center', ha='center', rotation=90, transform=ax.transAxes, fontweight='bold', fontsize=14)
            add_panel_label(ax, f"({labels[plot_idx-1]})")
            plot_idx += 1
            
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
    fig.colorbar(im, cax=cbar_ax, extend='both').set_label('Bias relative to MSWEP (mm day⁻¹)', fontsize=12)
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    save_fig(fig, "Fig_LMDZ_Seasonal_Bias_Maps")

# --- Figure 3: Regional annual cycle ---
def plot_figure_3(data):
    print("Generating Main Figure 3 (Regional Annual Cycles)...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    labels = list(string.ascii_lowercase)
    
    for i, region in enumerate(REGIONS):
        ax = axes[i]
        for model in MODEL_ORDER:
            ds = data[model]["precipitation"]
            mask = get_region_mask(ds, region)
            regional_mean = ds.where(mask).mean(dim=['lat', 'lon'])
            monthly_clim = regional_mean.groupby('time.month').mean()
            
            kwargs = {'label': model, 'color': MODEL_COLORS[model], 'linewidth': 2}
            if model == 'MSWEP':
                kwargs.update({'color': 'black', 'linewidth': 3, 'marker': 'o'})
            elif model in ['LMDZ250', 'LMDZ35']:
                kwargs.update({'linestyle': '--', 'alpha': 0.8})
                
            ax.plot(range(1, 13), monthly_clim, **kwargs)
            
        ax.set_title(region.replace('_', ' ').title(), fontweight='bold')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
        if i % 2 == 0: ax.set_ylabel('Precipitation (mm day⁻¹)')
        ax.grid(True, linestyle='--', alpha=0.5)
        add_panel_label(ax, f"({labels[i]})")
        
        if region == "south":
            ax.text(0.5, 0.9, 'Note different y-axis scale', ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray', style='italic')

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=12)
    
    plt.tight_layout()
    save_fig(fig, "Fig_LMDZ_Regional_Annual_Cycles")

# --- Figure 4: High-resolution signal recovery ---
def plot_figure_4():
    print("Generating Main Figure 4 (Increment Maps)...")
    lmdz35_native = load_standardized_lmdz(LMDZ35_NATIVE_PATH)
    lmdz250_native = load_standardized_lmdz(LMDZ250_NATIVE_PATH)
    
    unet_10km = xr.open_dataset(os.path.join(PRED_ROOT, "unet_exp32_parallel/unet_exp32_lmdz_250_present_true.nc"))[["precipitation"]].sel(time=slice("1979-01-01", "2014-12-31"))
    cnn_10km = xr.open_dataset(os.path.join(PRED_ROOT, "cnn_exp5/cnn_lmdz_250_present_true.nc"))[["precipitation"]].sel(time=slice("1979-01-01", "2014-12-31"))
    vit_10km = xr.open_dataset(os.path.join(PRED_ROOT, "vit_precip_exp21_best_hybrid/vit_lmdz_250_present_true.nc"))[["precipitation"]].sel(time=slice("1979-01-01", "2014-12-31"))

    # Interpolate outputs to native LMDZ35 grid
    lmdz250_on_35 = lmdz250_native.interp(lat=lmdz35_native.lat, lon=lmdz35_native.lon, method='linear')
    unet_on_35 = unet_10km.interp(lat=lmdz35_native.lat, lon=lmdz35_native.lon, method='linear')
    cnn_on_35 = cnn_10km.interp(lat=lmdz35_native.lat, lon=lmdz35_native.lon, method='linear')
    vit_on_35 = vit_10km.interp(lat=lmdz35_native.lat, lon=lmdz35_native.lon, method='linear')
    
    # Calculate increments
    mean_35 = lmdz35_native["precipitation"].mean(dim='time')
    mean_250 = lmdz250_on_35["precipitation"].mean(dim='time')
    mean_unet = unet_on_35["precipitation"].mean(dim='time')
    mean_cnn = cnn_on_35["precipitation"].mean(dim='time')
    mean_vit = vit_on_35["precipitation"].mean(dim='time')
    
    mask = get_region_mask(mean_35, "allmorr")
    
    delta_dyn = (mean_35 - mean_250).where(mask)
    delta_unet = (mean_unet - mean_250).where(mask)
    delta_cnn = (mean_cnn - mean_250).where(mask)
    delta_vit = (mean_vit - mean_250).where(mask)
    
    residual_unet = delta_unet - delta_dyn
    residual_cnn = delta_cnn - delta_dyn
    residual_vit = delta_vit - delta_dyn
    
    fig = plt.figure(figsize=(20, 10))
    labels = list(string.ascii_lowercase)
    
    panels = [
        (1, delta_dyn, r"Dynamical increment: $\Delta_{dyn}$"),
        (2, delta_unet, r"U-Net increment: $\Delta_{U-Net}$"),
        (3, delta_cnn, r"CNN increment: $\Delta_{CNN}$"),
        (4, delta_vit, r"ViT increment: $\Delta_{ViT}$"),
        (6, residual_unet, r"U-Net residual: $\Delta_{U-Net} - \Delta_{dyn}$"),
        (7, residual_cnn, r"CNN residual: $\Delta_{CNN} - \Delta_{dyn}$"),
        (8, residual_vit, r"ViT residual: $\Delta_{ViT} - \Delta_{dyn}$")
    ]
    
    for i, (subplot_idx, data_da, title) in enumerate(panels):
        ax = fig.add_subplot(2, 4, subplot_idx, projection=ccrs.PlateCarree())
        setup_map_axis(ax)
        im = ax.pcolormesh(data_da.lon, data_da.lat, data_da, transform=ccrs.PlateCarree(), cmap=cmap_bias, norm=norm_inc)
        
        ax.set_title(title, loc='center', fontweight='bold', fontsize=11)
        add_panel_label(ax, f"({labels[i]})")
    
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both').set_label('Precipitation increment (mm day⁻¹)', fontsize=12)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    save_fig(fig, "Fig_LMDZ_HighRes_Increment_Recovery")
    
    # Save S4: Full Scale if needed. Actually it's just the same plot without the norm limitation
    fig_s4 = plt.figure(figsize=(20, 10))
    # Full range colormap
    for i, (subplot_idx, data_da, title) in enumerate(panels):
        ax = fig_s4.add_subplot(2, 4, subplot_idx, projection=ccrs.PlateCarree())
        setup_map_axis(ax)
        vmax = max(abs(float(data_da.min())), abs(float(data_da.max())))
        if vmax == 0 or np.isnan(vmax): vmax = 1
        im_s4 = ax.pcolormesh(data_da.lon, data_da.lat, data_da, transform=ccrs.PlateCarree(), cmap=cmap_bias, vmin=-vmax, vmax=vmax)
        
        ax.set_title(title, loc='center', fontweight='bold', fontsize=11)
        add_panel_label(ax, f"({labels[i]})")
        # Add colorbar per plot to show full range
        fig_s4.colorbar(im_s4, ax=ax, orientation='horizontal', pad=0.05)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    save_fig(fig_s4, "FigS_LMDZ_FullScale_Increment")
    
    return delta_dyn, delta_unet, delta_cnn, delta_vit

# --- Figure 5: Spatial pattern agreement ---
def plot_figure_5(delta_dyn, delta_unet, delta_cnn, delta_vit):
    print("Generating Main Figure 5 (Spatial Pattern Agreement)...")
    fig = plt.figure(figsize=(10, 6))
    ax_bar = fig.add_subplot(1, 1, 1)
    
    corrs_unet, corrs_cnn, corrs_vit = [], [], []
    reg_labels = ["Whole Domain"] + [r.replace('_', ' ').title() for r in REGIONS]
    
    for reg in ["allmorr"] + REGIONS:
        rmask = get_region_mask(delta_dyn, reg)
        dd = delta_dyn.where(rmask).values.flatten()
        du = delta_unet.where(rmask).values.flatten()
        dc = delta_cnn.where(rmask).values.flatten()
        dv = delta_vit.where(rmask).values.flatten()
        
        valid = ~np.isnan(dd) & ~np.isnan(du) & ~np.isnan(dc) & ~np.isnan(dv)
        corrs_unet.append(np.corrcoef(dd[valid], du[valid])[0,1] if np.sum(valid)>10 else np.nan)
        corrs_cnn.append(np.corrcoef(dd[valid], dc[valid])[0,1] if np.sum(valid)>10 else np.nan)
        corrs_vit.append(np.corrcoef(dd[valid], dv[valid])[0,1] if np.sum(valid)>10 else np.nan)
        
    x = np.arange(len(reg_labels))
    width = 0.25
    ax_bar.bar(x - width, corrs_unet, width, label='U-Net', color=MODEL_COLORS['U-Net'])
    ax_bar.bar(x, corrs_cnn, width, label='CNN', color=MODEL_COLORS['CNN'])
    ax_bar.bar(x + width, corrs_vit, width, label='ViT', color=MODEL_COLORS['ViT'])
    
    ax_bar.axhline(0.5, color='gray', linestyle='--', linewidth=1.5)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel('Spatial correlation (r)')
    ax_bar.set_title("Spatial pattern agreement", fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(reg_labels, rotation=45, ha='right')
    ax_bar.legend()
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
    
    def add_vals(corrs, offset):
        for i, v in enumerate(corrs):
            ax_bar.text(i + offset, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
            
    add_vals(corrs_unet, -width)
    add_vals(corrs_cnn, 0)
    add_vals(corrs_vit, width)
        
    plt.tight_layout()
    save_fig(fig, "Fig_LMDZ_Spatial_Pattern_Agreement")

# --- Supplementary Figure S1 & S2: Metrics ---
def compute_regional_metrics(data):
    models = ["LMDZ250", "LMDZ35", "U-Net", "CNN", "ViT"]
    metrics = {"Bias": np.nan, "RMSE": np.nan, "Correlation": np.nan, "R95 Error": np.nan, "CDD Error": np.nan}
    results = {reg: {mod: metrics.copy() for mod in models} for reg in REGIONS}
    
    mswep = data["MSWEP"]["precipitation"]
    for reg in REGIONS:
        mask = get_region_mask(mswep, reg)
        ref_masked = mswep.where(mask)
        ref_mean = ref_masked.mean(dim=['lat', 'lon'])
        ref_cdd = calc_cdd(ref_mean)
        p95 = ref_mean.quantile(0.95)
        f_ref = (ref_mean > p95).sum().values / len(ref_mean) * 365.25
        
        for model in models:
            ts_mod = data[model]["precipitation"].where(mask).mean(dim=['lat', 'lon'])
            
            results[reg][model]["Bias"] = float((ts_mod.mean() - ref_mean.mean()).values)
            results[reg][model]["RMSE"] = float(np.sqrt(((ts_mod - ref_mean)**2).mean()).values)
            v_mod = ts_mod.values.flatten()
            v_ref = ref_mean.values.flatten()
            min_len = min(len(v_mod), len(v_ref))
            results[reg][model]["Correlation"] = float(np.corrcoef(v_mod[:min_len], v_ref[:min_len])[0,1])
            f_mod = (ts_mod > p95).sum().values / len(ts_mod) * 365.25
            results[reg][model]["R95 Error"] = float(f_mod - f_ref)
            results[reg][model]["CDD Error"] = float(calc_cdd(ts_mod) - ref_cdd)
            
    return results, models

def plot_figure_S1_S2(data):
    print("Generating Figures S1 and S2 (Metrics Heatmaps)...")
    results, models = compute_regional_metrics(data)
    y_labels = [r.replace('_', ' ').title() for r in REGIONS]
    
    # --- Figure S1: Mean Metrics ---
    fig_s1, axes_s1 = plt.subplots(1, 3, figsize=(18, 5))
    labels = ['(a)', '(b)', '(c)']
    metric_names = {"Bias": "Bias (mm day⁻¹)", "RMSE": "RMSE (mm day⁻¹)", "Correlation": "Daily Temporal Correlation (r)"}
    
    for i, metric in enumerate(["Bias", "RMSE", "Correlation"]):
        df = pd.DataFrame({m: [results[r][m][metric] for r in REGIONS] for m in models}, index=y_labels)
        
        if metric == "Bias":
            vmax = np.max(np.abs(df.values))
            sns.heatmap(df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes_s1[i], vmin=-vmax, vmax=vmax, cbar_kws={'label': 'mm day⁻¹'})
        elif metric == "RMSE":
            sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", ax=axes_s1[i], cbar_kws={'label': 'mm day⁻¹'})
        else:
            sns.heatmap(df, annot=True, fmt=".2f", cmap="magma", ax=axes_s1[i], vmin=0, vmax=1)
            
        axes_s1[i].set_title(metric_names[metric], fontweight='bold')
        add_panel_label(axes_s1[i], labels[i])
        
    plt.tight_layout()
    save_fig(fig_s1, "FigS_LMDZ_Mean_Metrics")
    
    # --- Figure S2: Extreme Metrics ---
    fig_s2, axes_s2 = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['(a)', '(b)']
    metric_names_s2 = {"R95 Error": "R95 Event-frequency Error (days)", "CDD Error": "CDD Error (days)"}
    
    for i, metric in enumerate(["R95 Error", "CDD Error"]):
        df = pd.DataFrame({m: [results[r][m][metric] for r in REGIONS] for m in models}, index=y_labels)
        vmax = np.max(np.abs(df.values))
        sns.heatmap(df, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=axes_s2[i], vmin=-vmax, vmax=vmax, cbar_kws={'label': 'days'})
        axes_s2[i].set_title(f"{metric_names_s2[metric]}\nSigned Error (Model - MSWEP)", fontweight='bold')
        add_panel_label(axes_s2[i], labels[i])
        
    plt.tight_layout()
    save_fig(fig_s2, "FigS_LMDZ_Extreme_Metrics")

# --- Supplementary Figure S3: Distributional Comparison ---
def plot_figure_S3(data):
    print("Generating Figure S3 (Distribution Boxplots)...")
    # Actually this is S3 in the filename list given by the user at the end:
    # "FigS_LMDZ_Distributional_Comparison.png"
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), sharex=True)
    
    for i, reg in enumerate(REGIONS):
        mask = get_region_mask(data["MSWEP"]["precipitation"].mean('time'), reg)
        for j, (season, months) in enumerate(SEASONS.items()):
            ax = axes[i, j]
            plot_data = []
            for model in MODEL_ORDER:
                ds = data[model]["precipitation"]
                ds_reg = ds.where(mask).mean(dim=['lat', 'lon'])
                # Monthly mean of daily precipitation over the whole period
                ds_month = ds_reg.resample(time='1M').mean()
                ds_season = ds_month.sel(time=ds_month.time.dt.month.isin(months))
                plot_data.append(ds_season.values.flatten())
                
            bp = ax.boxplot(plot_data, patch_artist=True,
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(color='black'), capprops=dict(color='black'))
            
            # Set colors
            for patch, model in zip(bp['boxes'], MODEL_ORDER):
                patch.set_facecolor(MODEL_COLORS[model])
                patch.set_alpha(0.7)
            
            # Update tick labels
            ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
            ax.set_xticklabels(MODEL_ORDER, rotation=45)
            
            if i == 0: ax.set_title(season, fontweight='bold')
            if j == 0: ax.set_ylabel(f"{reg.replace('_', ' ').title()}\nPrecip (mm day⁻¹)", fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
    plt.tight_layout()
    save_fig(fig, "FigS_LMDZ_Distributional_Comparison")

def generate_all():
    data = load_10km_data()
    plot_figure_1(data)
    plot_figure_2(data)
    plot_figure_3(data)
    delta_dyn, delta_unet, delta_cnn, delta_vit = plot_figure_4()
    plot_figure_5(delta_dyn, delta_unet, delta_cnn, delta_vit)
    plot_figure_S1_S2(data)
    plot_figure_S3(data) # Boxplots
    print("All figures generated successfully in results/figures/lmdz_paper/")

if __name__ == "__main__":
    generate_all()
