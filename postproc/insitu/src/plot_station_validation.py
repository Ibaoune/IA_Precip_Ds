"""
Author: M. El Aabaribaoune (@um6p)
Description: Part of the post-processing and evaluation pipeline for the downscaling project.
"""

#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import string
import warnings

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

warnings.filterwarnings('ignore')

# Add paths
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import utils
import insitu_utils

# --- Configuration ---
STATIONS = ["CASABLANCA", "FES", "BGE TANGER MED", "DAKHLA", "AGADIR"]
MODELS = ["MSWEP", "GLM", "U-Net", "CNN", "ViT"]
MODEL_COLORS = {
    'MSWEP': '#000000',
    'GLM': '#F4A261', # Mapping to LMDZ250
    'U-Net': '#1D3557',
    'CNN': '#457B9D',
    'ViT': '#E63946'
}

START_DATE = "1979-01-01"
END_DATE = "2014-12-31"

OUT_DIR = os.path.join(root_path, "results/figures/lmdz_paper")
os.makedirs(OUT_DIR, exist_ok=True)

def get_model_datasets(config):
    models_data = {}
    
    # Reference
    ref_ds = xr.open_dataset(os.path.join(root_path, config['reference']['file_path']))
    models_data['MSWEP'] = {"ds": ref_ds, "var_name": config['reference']['variable_name'], "color": "black"}

    # Others
    for d in config['datasets']:
        name = d['name']
        # Map LMDZ250 to GLM as per user request
        if name == 'LMDZ250':
            name = 'GLM'
        if name in MODELS:
            fpath = os.path.join(root_path, d['file_path'])
            if os.path.exists(fpath):
                ds_mod = xr.open_dataset(fpath)
                models_data[name] = {"ds": ds_mod, "var_name": d['variable_name'], "color": MODEL_COLORS[name]}
                
    return models_data

def compute_metrics(obs, sim, threshold=1.0):
    """Compute deterministic and categorical metrics."""
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    obs = obs[valid]
    sim = sim[valid]
    
    if len(obs) == 0:
        return {k: np.nan for k in ["Bias", "RMSE", "MAE", "Correlation", "WetDayFreqBias", "POD", "FAR", "FBI", "R95Error", "CDDError"]}
    
    bias = np.mean(sim - obs)
    rmse = np.sqrt(np.mean((sim - obs)**2))
    mae = np.mean(np.abs(sim - obs))
    corr = np.corrcoef(obs, sim)[0, 1] if len(obs) > 1 else np.nan
    
    # Categorical
    obs_wet = obs >= threshold
    sim_wet = sim >= threshold
    
    hits = np.sum(obs_wet & sim_wet)
    misses = np.sum(obs_wet & ~sim_wet)
    false_alarms = np.sum(~obs_wet & sim_wet)
    
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    fbi = (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else np.nan
    
    # Wet day frequency bias (days per year)
    years = len(obs) / 365.25
    obs_wet_freq = np.sum(obs_wet) / years
    sim_wet_freq = np.sum(sim_wet) / years
    wet_freq_bias = sim_wet_freq - obs_wet_freq
    
    # R95 Event frequency error
    if np.sum(obs_wet) > 0:
        r95_thresh = np.percentile(obs[obs_wet], 95)
        f_obs = np.sum(obs > r95_thresh) / years
        f_sim = np.sum(sim > r95_thresh) / years
        r95_error = f_sim - f_obs
    else:
        r95_error = np.nan
        
    # CDD Error
    def get_cdd(arr):
        dry = (arr < threshold).astype(int)
        max_cdd = 0
        current_cdd = 0
        for val in dry:
            if val == 1:
                current_cdd += 1
                max_cdd = max(max_cdd, current_cdd)
            else:
                current_cdd = 0
        return max_cdd
    
    # Yearly CDD
    years_arr = np.repeat(np.arange(int(np.ceil(years))), 366)[:len(obs)]
    df = pd.DataFrame({'obs': obs, 'sim': sim, 'year': years_arr})
    cdd_obs = df.groupby('year')['obs'].apply(lambda x: get_cdd(x.values)).mean()
    cdd_sim = df.groupby('year')['sim'].apply(lambda x: get_cdd(x.values)).mean()
    cdd_error = cdd_sim - cdd_obs
    
    return {
        "Bias": bias,
        "RMSE": rmse,
        "MAE": mae,
        "Correlation": corr,
        "WetDayFreqBias": wet_freq_bias,
        "POD": pod,
        "FAR": far,
        "FBI": fbi,
        "R95Error": r95_error,
        "CDDError": cdd_error
    }

def generate_metadata_table(obs_df, stations_meta):
    results = []
    for station in STATIONS:
        if station not in stations_meta:
            continue
            
        st_data = obs_df[obs_df["Station"] == station]
        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]
        elev = st_data["Elevation"].iloc[0] if "Elevation" in st_data.columns else np.nan
        
        start_date = st_data["Date"].min()
        end_date = st_data["Date"].max()
        
        full_range = pd.date_range(start_date, end_date)
        actual_days = len(st_data["Precipitation"].dropna())
        missing_pct = (1 - actual_days / len(full_range)) * 100
        
        yearly_precip = st_data.groupby(st_data["Date"].dt.year)["Precipitation"].sum()
        mean_annual = yearly_precip.mean()
        
        yearly_wet = st_data[st_data["Precipitation"] >= 1.0].groupby(st_data["Date"].dt.year)["Precipitation"].count()
        wet_freq = yearly_wet.mean()
        
        # Region approximation based on latitude/longitude roughly
        if lat > 34 and lon > -6:
            region = "North-East"
        elif lat > 34:
            region = "North"
        elif lat < 30:
            region = "South"
        else:
            region = "West/Central"
            
        results.append({
            "Station": station,
            "Region": region,
            "Latitude": f"{lat:.3f}",
            "Longitude": f"{lon:.3f}",
            "Elevation (m)": f"{elev:.0f}" if not np.isnan(elev) else "N/A",
            "Climate setting": "To be filled",
            "Available period": f"{start_date.year}-{end_date.year}",
            "Missing data (%)": f"{missing_pct:.1f}%",
            "Mean annual precipitation (mm)": f"{mean_annual:.1f}",
            "Wet-day frequency (days/yr)": f"{wet_freq:.1f}"
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "Table_Station_Metadata.csv"), index=False)
    print("Metadata table saved.")
    return df

def plot_main_figure(metrics_df, stations_meta):
    print("Generating Main Figure (Station Map and Heatmaps)...")
    fig = plt.figure(figsize=(16, 12))
    
    # Setup subplots with gridspec
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- (a) Map ---
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
    ax_map.set_extent([-18, 0, 21, 37], crs=ccrs.PlateCarree())
    
    # Overlay Morocco boundary
    try:
        shapefile = utils.get_shapefile("allmorr")
        morocco = gpd.read_file(shapefile).to_crs("EPSG:4326").dissolve()
        ax_map.add_geometries(morocco.geometry, crs=ccrs.PlateCarree(), edgecolor='black', facecolor='whitesmoke', linewidth=1)
    except:
        pass
        
    # Plot stations
    for station in STATIONS:
        if station in stations_meta:
            lat = stations_meta[station]['lat']
            lon = stations_meta[station]['lon']
            ax_map.plot(lon, lat, marker='o', color='red', markersize=8, markeredgecolor='black', transform=ccrs.PlateCarree())
            ax_map.text(lon + 0.3, lat, station.title(), transform=ccrs.PlateCarree(), fontsize=10, fontweight='bold', va='center')
            
    ax_map.set_title("Station Locations", fontweight='bold')
    ax_map.text(-0.05, 1.05, "(a)", transform=ax_map.transAxes, fontsize=14, fontweight='bold')
    
    # --- Prepare heatmap data ---
    models_plot = [m for m in MODELS if m in metrics_df['Model'].unique()]
    
    def prep_heatmap(metric):
        pivot = metrics_df.pivot(index="Station", columns="Model", values=metric)
        # Reorder columns
        pivot = pivot[[m for m in models_plot if m in pivot.columns]]
        # Clean station names
        pivot.index = [s.title() for s in pivot.index]
        return pivot

    # --- (b) RMSE Heatmap ---
    ax_rmse = fig.add_subplot(gs[0, 1])
    rmse_data = prep_heatmap("RMSE")
    sns.heatmap(rmse_data, annot=True, fmt=".1f", cmap="viridis", ax=ax_rmse, cbar_kws={'label': 'RMSE (mm day⁻¹)'})
    ax_rmse.set_title("Station-wise Mean Performance (RMSE)", fontweight='bold')
    ax_rmse.set_ylabel("")
    ax_rmse.set_xlabel("")
    ax_rmse.text(-0.05, 1.05, "(b)", transform=ax_rmse.transAxes, fontsize=14, fontweight='bold')
    
    # --- (c) Correlation Heatmap ---
    ax_corr = fig.add_subplot(gs[1, 0])
    corr_data = prep_heatmap("Correlation")
    sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="magma", vmin=0, vmax=1, ax=ax_corr, cbar_kws={'label': 'Correlation (r)'})
    ax_corr.set_title("Station-wise Temporal Correlation", fontweight='bold')
    ax_corr.set_ylabel("")
    ax_corr.set_xlabel("")
    ax_corr.text(-0.05, 1.05, "(c)", transform=ax_corr.transAxes, fontsize=14, fontweight='bold')
    
    # --- (d) Wet-day Frequency Bias ---
    ax_wet = fig.add_subplot(gs[1, 1])
    wet_data = prep_heatmap("WetDayFreqBias")
    vmax = np.nanmax(np.abs(wet_data.values))
    sns.heatmap(wet_data, annot=True, fmt=".1f", cmap="RdBu_r", vmin=-vmax, vmax=vmax, center=0, ax=ax_wet, cbar_kws={'label': 'Bias (days/year)'})
    ax_wet.set_title("Wet-day Frequency Bias (Model - Station)", fontweight='bold')
    ax_wet.set_ylabel("")
    ax_wet.set_xlabel("")
    ax_wet.text(-0.05, 1.05, "(d)", transform=ax_wet.transAxes, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig_Station_Validation.png"), dpi=400, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, "Fig_Station_Validation.pdf"), bbox_inches='tight')
    plt.close()
    
def plot_annual_cycles(obs_df, stations_meta, models_data):
    print("Generating Supplementary Figure S1 (Annual Cycles)...")
    num_stations = len(STATIONS)
    ncols = min(3, num_stations)
    nrows = int(np.ceil(num_stations / 3))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), dpi=300)
    axes_flat = axes.flatten() if num_stations > 1 else [axes]
    
    for idx in range(num_stations, len(axes_flat)):
        axes_flat[idx].set_visible(False)
        
    months = np.arange(1, 13)
    
    for i, station in enumerate(STATIONS):
        if station not in stations_meta:
            continue
            
        ax = axes_flat[i]
        st_obs = obs_df[obs_df["Station"] == station].copy().sort_values("Date")
        obs_series = st_obs.set_index("Date")["Precipitation"]
        
        obs_clim = st_obs.groupby(st_obs["Date"].dt.month)["Precipitation"].mean().reindex(months)
        ax.plot(months, obs_clim.values, label="Station Observations", color="black", linewidth=3, linestyle="-")
        
        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]
        
        for name in MODELS:
            if name not in models_data: continue
            info = models_data[name]
            mod_da = insitu_utils.extract_nearest_gridpoint(info["ds"], lat, lon, info["var_name"])
            aligned = insitu_utils.align_series(obs_series, mod_da)
            
            if len(aligned) > 0:
                aligned["month"] = aligned.index.month
                sim_clim = aligned.groupby("month")["sim"].mean().reindex(months)
                ax.plot(months, sim_clim.values, label=name, color=info["color"], linewidth=1.5)
                
        ax.set_title(station.title(), fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_ylabel("Precipitation (mm day⁻¹)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02 + 0.03 * nrows))
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "FigS_Station_Annual_Cycles.png"), dpi=300, bbox_inches='tight')
    plt.close()

def compute_qq_quantiles(series, n_quantiles=100, threshold=1.0):
    vals = series.values
    vals = vals[~np.isnan(vals)]
    vals = vals[vals >= threshold]
    if len(vals) == 0: return np.full(n_quantiles, np.nan)
    vals_sorted = np.sort(vals)
    indices = np.linspace(0, len(vals_sorted) - 1, n_quantiles).astype(int)
    return vals_sorted[indices]

def plot_qq_plots(obs_df, stations_meta, models_data):
    print("Generating Supplementary Figure S2 (QQ Plots)...")
    num_stations = len(STATIONS)
    ncols = min(3, num_stations)
    nrows = int(np.ceil(num_stations / 3))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=300)
    axes_flat = axes.flatten() if num_stations > 1 else [axes]
    
    for idx in range(num_stations, len(axes_flat)):
        axes_flat[idx].set_visible(False)
        
    for i, station in enumerate(STATIONS):
        if station not in stations_meta: continue
            
        ax = axes_flat[i]
        st_obs = obs_df[obs_df["Station"] == station].copy().sort_values("Date")
        obs_series = st_obs.set_index("Date")["Precipitation"]
        
        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]
        
        obs_quantiles = compute_qq_quantiles(obs_series, threshold=1.0)
        max_val = np.nanmax(obs_quantiles) if not np.isnan(obs_quantiles).all() else 10
        
        for name in MODELS:
            if name not in models_data: continue
            info = models_data[name]
            mod_da = insitu_utils.extract_nearest_gridpoint(info["ds"], lat, lon, info["var_name"])
            aligned = insitu_utils.align_series(obs_series, mod_da)
            
            if len(aligned) > 0:
                mod_quantiles = compute_qq_quantiles(aligned["sim"], threshold=1.0)
                ax.scatter(obs_quantiles, mod_quantiles, label=name, color=info["color"], s=15, alpha=0.7)
                curr_max = np.nanmax(mod_quantiles)
                if not np.isnan(curr_max): max_val = max(max_val, curr_max)
                
        ax.plot([1.0, max_val], [1.0, max_val], 'k--', alpha=0.5, label="1:1 line")
        
        ax.set_title(station.title(), fontweight='bold')
        ax.set_xlabel("Station Observations (mm day⁻¹)")
        ax.set_ylabel("Gridded Estimate (mm day⁻¹)")
        
        # Use log scale to handle skewness
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        ax.set_xlim(1, max_val * 1.1)
        ax.set_ylim(1, max_val * 1.1)
        ax.grid(True, linestyle="--", alpha=0.5)
        
    handles, labels = axes_flat[0].get_legend_handles_labels()
    # deduplicate labels
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=len(by_label), bbox_to_anchor=(0.5, 1.02 + 0.03 * nrows))
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "FigS_Station_QQPlots.png"), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../insitu/config.yaml"))
    config = utils.load_config(config_path)
    
    # 1. Load Station Observations
    obs_df = insitu_utils.load_insitu_observations(start_date=START_DATE, end_date=END_DATE)
    stations_meta = insitu_utils.get_station_metadata(obs_df)
    
    # Export Metadata Table
    generate_metadata_table(obs_df, stations_meta)
    
    # 2. Load Models
    print("Loading models...")
    models_data = get_model_datasets(config)
    
    # 3. Compute Metrics
    all_metrics = []
    print("Extracting time series and computing metrics...")
    for station in STATIONS:
        if station not in stations_meta: continue
        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]
        
        st_obs = obs_df[obs_df["Station"] == station].copy().sort_values("Date")
        obs_series = st_obs.set_index("Date")["Precipitation"]
        
        for name in MODELS:
            if name not in models_data: continue
            info = models_data[name]
            
            mod_da = insitu_utils.extract_nearest_gridpoint(info["ds"], lat, lon, info["var_name"])
            aligned = insitu_utils.align_series(obs_series, mod_da)
            
            if len(aligned) > 0:
                metrics = compute_metrics(aligned["obs"].values, aligned["sim"].values)
                metrics["Station"] = station
                metrics["Model"] = name
                all_metrics.append(metrics)
                
    metrics_df = pd.DataFrame(all_metrics)
    
    # Arrange columns
    cols = ["Station", "Model", "Bias", "RMSE", "MAE", "Correlation", "WetDayFreqBias", "POD", "FAR", "FBI", "R95Error", "CDDError"]
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
    
    # Export Full Table S3
    metrics_df.to_csv(os.path.join(OUT_DIR, "TableS3_Station_Metrics.csv"), index=False)
    print("Saved TableS3_Station_Metrics.csv")
    
    # 4. Generate Figures
    plot_main_figure(metrics_df, stations_meta)
    plot_annual_cycles(obs_df, stations_meta, models_data)
    plot_qq_plots(obs_df, stations_meta, models_data)
    
    print("All validation outputs generated successfully.")

if __name__ == "__main__":
    main()
