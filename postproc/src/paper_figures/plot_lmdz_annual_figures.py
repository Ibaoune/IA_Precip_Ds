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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string

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

from src.utils import get_shapefile, BIAS_RDBU_WHITE

# --- Configuration ---
PRED_ROOT = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/inference/results/1979_2014"
DATASETS_DIR = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/data/datasets/datasets"
MSWEP_PATH = "/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc"

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

# --- Custom Colormaps ---
cmap_precip = mcolors.LinearSegmentedColormap.from_list("custom_precip", ["white", "lightblue", "darkblue", "darkgreen", "lightgreen", "yellow", "orange", "red"], N=256)
norm_precip = mcolors.BoundaryNorm([0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 10], ncolors=cmap_precip.N, extend='max')

cmap_bias = mcolors.LinearSegmentedColormap.from_list("custom_bias", BIAS_RDBU_WHITE, N=13)
norm_bias = mcolors.BoundaryNorm([-5, -3, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 3, 5], ncolors=cmap_bias.N, extend='both')

# --- Helper Functions ---
def add_panel_label(ax, text):
    ax.text(-0.05, 1.05, text, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='right')

def setup_map_axis(ax):
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':', edgecolor='lightgray')
    ax.set_extent([-18, 0, 21, 37], crs=ccrs.PlateCarree())
    
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

def load_10km_data():
    print("Loading 10km datasets...")
    mswep = xr.open_dataset(MSWEP_PATH)
    if "precipitation" not in mswep and "pr" in mswep: mswep = mswep.rename({"pr": "precipitation"})
    mswep = mswep[["precipitation"]].sel(time=slice("1979-01-01", "2014-12-31"))
    
    lmdz250_10km = xr.open_dataset(os.path.join(DATASETS_DIR, "raw_lmdz250_10km.nc"))[["precipitation"]]
    lmdz35_10km = xr.open_dataset(os.path.join(DATASETS_DIR, "raw_lmdz35_10km.nc"))[["precipitation"]]
    unet_10km = xr.open_dataset(os.path.join(PRED_ROOT, "unet_exp32_parallel/unet_exp32_parallel_lmdz_250_present_true.nc"))[["precipitation"]]
    cnn_10km = xr.open_dataset(os.path.join(PRED_ROOT, "cnn_exp5/cnn_lmdz_250_present_true.nc"))[["precipitation"]]
    vit_10km = xr.open_dataset(os.path.join(PRED_ROOT, "vit_precip_exp21_best_hybrid/vit_lmdz_250_present_true.nc"))[["precipitation"]]
    
    return {
        "MSWEP": mswep, "LMDZ250": lmdz250_10km, "LMDZ35": lmdz35_10km,
        "U-Net": unet_10km, "CNN": cnn_10km, "ViT": vit_10km
    }

def save_fig(fig, name):
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    print(f"Saved {name}")
    plt.close(fig)

# --- Figure 1: Annual Precipitation Maps ---
def plot_annual_figure_1(data):
    print("Generating Annual Precipitation Maps...")
    fig = plt.figure(figsize=(18, 8))
    labels = list(string.ascii_lowercase)
    
    mswep = data["MSWEP"]["precipitation"]
    mask_all = get_region_mask(mswep.mean(dim='time'), "allmorr")
    
    annual_data = {}
    for model in MODEL_ORDER:
        ds = data[model]["precipitation"]
        annual_data[model] = ds.mean(dim='time').where(mask_all)

    # 2 rows, 3 cols
    for j, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(2, 3, j+1, projection=ccrs.PlateCarree())
        setup_map_axis(ax)
        ds_mean = annual_data[model]
        
        im = ax.pcolormesh(ds_mean.lon, ds_mean.lat, ds_mean, transform=ccrs.PlateCarree(), cmap=cmap_precip, norm=norm_precip)
        
        ax.set_title(model, loc='center', fontweight='bold', fontsize=12)
        add_panel_label(ax, f"({labels[j]})")
            
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
    fig.colorbar(im, cax=cbar_ax, extend='max').set_label('Mean annual precipitation (mm day⁻¹)', fontsize=12)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    save_fig(fig, "Fig_LMDZ_Annual_Precipitation_Maps")

# --- Figure 2: Annual Bias Maps ---
def plot_annual_figure_2(data):
    print("Generating Annual Bias Maps...")
    fig = plt.figure(figsize=(20, 6))
    labels = list(string.ascii_lowercase)
    
    mswep = data["MSWEP"]["precipitation"]
    mask_all = get_region_mask(mswep.mean(dim='time'), "allmorr")
    mswep_mean = mswep.mean(dim='time').where(mask_all)
    
    annual_data = {}
    for model in MODEL_ORDER:
        ds = data[model]["precipitation"]
        annual_data[model] = ds.mean(dim='time').where(mask_all)

    plot_models = [m for m in MODEL_ORDER if m != "MSWEP"]
    
    for j, model in enumerate(plot_models):
        ax = fig.add_subplot(1, 5, j+1, projection=ccrs.PlateCarree())
        setup_map_axis(ax)
        ds_mean = annual_data[model]
        bias = ds_mean - mswep_mean
        
        im = ax.pcolormesh(bias.lon, bias.lat, bias, transform=ccrs.PlateCarree(), cmap=cmap_bias, norm=norm_bias)
        
        ax.set_title(f"{model} - MSWEP", loc='center', fontweight='bold', fontsize=12)
        add_panel_label(ax, f"({labels[j]})")
            
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])
    fig.colorbar(im, cax=cbar_ax, extend='both').set_label('Bias relative to MSWEP (mm day⁻¹)', fontsize=12)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    save_fig(fig, "Fig_LMDZ_Annual_Bias_Maps")

# --- Figure 3: Annual Distributional Comparison ---
def plot_annual_figure_S3(data):
    print("Generating Annual Distribution Boxplots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, reg in enumerate(REGIONS):
        mask = get_region_mask(data["MSWEP"]["precipitation"].mean('time'), reg)
        ax = axes[i]
        plot_data = []
        for model in MODEL_ORDER:
            ds = data[model]["precipitation"]
            ds_reg = ds.where(mask).mean(dim=['lat', 'lon'])
            # Annual mean of daily precipitation over the whole period
            ds_year = ds_reg.groupby('time.year').mean()
            plot_data.append(ds_year.values.flatten())
            
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
        
        ax.set_title(f"{reg.replace('_', ' ').title()}", fontweight='bold')
        ax.set_ylabel(f"Precipitation (mm day⁻¹)", fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
            
    plt.tight_layout()
    save_fig(fig, "FigS_LMDZ_Annual_Distributional_Comparison")

def generate_all():
    data = load_10km_data()
    plot_annual_figure_1(data)
    plot_annual_figure_2(data)
    plot_annual_figure_S3(data)
    print("All annual figures generated successfully in results/figures/lmdz_paper/")

if __name__ == "__main__":
    generate_all()
