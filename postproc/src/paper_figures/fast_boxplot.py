#!/usr/bin/env python3
"""
Fast 2x2 Annual Boxplot — reads each file ONCE, computes all regions in one pass.
"""
import os, sys
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/inference/results/1979_2014"
MSWEP = "/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc"
DATA_DIR = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/data/datasets/datasets"

MODELS = {
    "MSWEP"   : MSWEP,
    "LMDZ250" : os.path.join(DATA_DIR, "raw_lmdz250_10km.nc"),
    "LMDZ35"  : os.path.join(DATA_DIR, "raw_lmdz35_10km.nc"),
    "U-Net"   : os.path.join(ROOT, "unet_exp32_parallel/unet_exp32_lmdz_250_present_true.nc"),
    "CNN"     : os.path.join(ROOT, "cnn_exp5/cnn_lmdz_250_present_true.nc"),
    "ViT"     : os.path.join(ROOT, "vit_precip_exp21_best_hybrid/vit_lmdz_250_present_true.nc"),
}
COLORS = {
    'MSWEP':'#000000','LMDZ250':'#F4A261','LMDZ35':'#E76F51',
    'U-Net':'#1D3557','CNN':'#457B9D','ViT':'#E63946'
}
ORDER = list(MODELS.keys())

# Region boxes: (lat_min, lat_max, lon_min, lon_max)
REGIONS = {
    "North"      : (33.0, 37.0, -18.0, 0.0),
    "North-East" : (33.0, 37.0,  -3.0, 0.0),
    "East"       : (29.0, 33.0,  -3.0, 0.0),
    "South"      : (21.0, 29.0, -12.0, 0.0),
}

# results[model][region] = array of annual means
results = {m: {r: None for r in REGIONS} for m in MODELS}

for mname, mpath in MODELS.items():
    print(f"Reading {mname}...", flush=True)
    ds = nc.Dataset(mpath)

    # Find precip variable
    var = next((v for v in ['precipitation','pr','precip'] if v in ds.variables), None)
    if var is None:
        print(f"  WARNING: no precip var in {mpath}"); ds.close(); continue

    lats = ds.variables['lat' if 'lat' in ds.variables else 'latitude'][:]
    lons = ds.variables['lon' if 'lon' in ds.variables else 'longitude'][:]

    # Time — read years for filtering
    t_var = ds.variables['time']
    times = nc.num2date(t_var[:], t_var.units, only_use_cftime_datetimes=False)
    years = np.array([t.year for t in times])
    t_mask = (years >= 1979) & (years <= 2014)
    t_idx  = np.where(t_mask)[0]
    years_sel = years[t_idx]

    # Read ALL spatial data once (full domain, subset time)
    # Use slice for time to avoid slow fancy indexing
    t_start, t_end = t_idx[0], t_idx[-1] + 1
    data = ds.variables[var][t_start:t_end]  # shape: (T, lat, lon) — fast sequential read
    if hasattr(data, 'filled'):
        data = data.filled(np.nan)

    # Scale if needed (kg/m2/s → mm/day)
    if np.nanmean(data) < 0.01:
        data = data * 86400.0

    # Clip extremes (UNet-32 explosion safety)
    data = np.clip(data, 0, 500.0)

    ds.close()

    # Compute annual mean for each region (no more disk I/O)
    for rname, (lat_min, lat_max, lon_min, lon_max) in REGIONS.items():
        ilat = np.where((lats >= lat_min) & (lats <= lat_max))[0]
        ilon = np.where((lons >= lon_min) & (lons <= lon_max))[0]
        if len(ilat) == 0 or len(ilon) == 0:
            results[mname][rname] = np.array([np.nan]); continue

        reg_data = data[:, ilat[0]:ilat[-1]+1, ilon[0]:ilon[-1]+1]
        spatial_mean = np.nanmean(reg_data, axis=(1, 2))

        unique_yrs = np.unique(years_sel)
        annual = np.array([np.nanmean(spatial_mean[years_sel == y]) for y in unique_yrs])
        results[mname][rname] = annual

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Plotting...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()

for i, (rname, ax) in enumerate(zip(REGIONS, axes)):
    plot_data = [results[m][rname] for m in ORDER]
    bp = ax.boxplot(plot_data, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.8),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, m in zip(bp['boxes'], ORDER):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.78)
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels(ORDER, rotation=30, ha='right', fontsize=9)
    ax.set_title(rname, fontweight='bold', fontsize=11)
    ax.set_ylabel("Annual mean precip (mm day⁻¹)", fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

out = "results/inference_retained/figures/lmdz_paper/Fig_2x2_Annual_Boxplots.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches='tight')
print(f"Done → {out}")
