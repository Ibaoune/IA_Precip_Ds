import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Add postproc/src to sys.path
root_path = str(Path(__file__).resolve().parents[2])
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import utils
import insitu_utils

warnings.filterwarnings('ignore')

def compute_qq_quantiles(series, n_quantiles=1000, threshold=1.0):
    """
    Sorts and extracts n_quantiles from series values that exceed threshold.
    """
    vals = series.values
    vals = vals[~np.isnan(vals)]
    vals = vals[vals >= threshold]
    
    if len(vals) == 0:
        return np.full(n_quantiles, np.nan)
        
    vals_sorted = np.sort(vals)
    indices = np.linspace(0, len(vals_sorted) - 1, n_quantiles).astype(int)
    return vals_sorted[indices]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Default config points to config.yaml in the insitu directory
    default_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config.yaml"))
    parser.add_argument("--config", default=default_config, help="Path to config file")
    args, unknown = parser.parse_known_args()

    # === Load Configuration ===
    config = utils.load_config(args.config)
    
    params = config['parameters']
    ref_cfg = config['reference']
    datasets_cfg = config['datasets']
    obs_cfg = config.get('observations', {})

    # Set up results directories
    insitu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(insitu_root, "Figs")
    csv_results_dir = os.path.join(results_dir, "results")
    os.makedirs(csv_results_dir, exist_ok=True)

    # Threshold and quantile configuration
    threshold = config.get('metric', {}).get('threshold', 1.0)
    n_quantiles = config.get('metric', {}).get('n_quantiles', 1000)

    # === 1. Load Station Observations ===
    obs_df = insitu_utils.load_insitu_observations(
        excel_path=obs_cfg.get('excel_path', insitu_utils.DEFAULT_EXCEL_PATH),
        cache_path=obs_cfg.get('cache_path', insitu_utils.DEFAULT_CACHE_PATH),
        start_date=params['start_date'],
        end_date=params['end_date']
    )
    stations_meta = insitu_utils.get_station_metadata(obs_df)
    stations_list = params.get('stations', ["CASABLANCA", "FES", "BGE TANGER MED", "DAKHLA", "AGADIR"])
    stations_to_process = [s for s in stations_list if s in stations_meta]
    
    if not stations_to_process:
        print("[WARNING] Requested stations not found. Processing first 3 available.")
        stations_to_process = list(stations_meta.keys())[:3]

    # === 2. Load Model Datasets ===
    print("[INFO] Pre-loading models and predictions...")
    models_data = {}
    
    # Add reference dataset (e.g. MSWEP)
    ref_ds = xr.open_dataset(os.path.join(root_path, ref_cfg['file_path']))
    models_data[ref_cfg['name'].upper()] = {
        "ds": ref_ds,
        "var_name": ref_cfg['variable_name'],
        "color": "black"
    }

    # Add downscaled predictions
    colors_palette = ["red", "darkblue", "green", "lightblue", "orange", "purple"]
    
    for idx, d in enumerate(datasets_cfg):
        fpath = os.path.join(root_path, d['file_path'])
        if os.path.exists(fpath):
            ds_mod = xr.open_dataset(fpath)
            models_data[d['name'].upper()] = {
                "ds": ds_mod,
                "var_name": d['variable_name'],
                "color": colors_palette[idx % len(colors_palette)]
            }
        else:
            print(f"[WARNING] Prediction file not found: {fpath}. Skipping.")

    # Initialize plotting grid (max 3 stations per line)
    num_stations = len(stations_to_process)
    ncols = min(3, num_stations)
    nrows = int(np.ceil(num_stations / 3))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=300)
    
    # Flatten and wrap axes list
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    # Hide any unused axes in the grid
    for idx in range(num_stations, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for i, station in enumerate(stations_to_process):
        ax = axes_flat[i]
        st_obs = obs_df[obs_df["Station"] == station].copy().sort_values("Date")
        obs_series = st_obs.set_index("Date")["Precipitation"]
        
        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]
        
        # Calculate quantiles for OBS
        obs_quantiles = compute_qq_quantiles(obs_series, n_quantiles=n_quantiles, threshold=threshold)
        
        results = {
            "quantile_rank": np.linspace(0, 1, n_quantiles),
            "OBS": obs_quantiles
        }
        
        # Calculate quantiles for models
        for name, info in models_data.items():
            mod_da = insitu_utils.extract_nearest_gridpoint(info["ds"], lat, lon, info["var_name"])
            aligned = insitu_utils.align_series(obs_series, mod_da)
            
            if len(aligned) > 0:
                mod_quantiles = compute_qq_quantiles(aligned["sim"], n_quantiles=n_quantiles, threshold=threshold)
                results[name] = mod_quantiles
                
                # Plot
                ax.scatter(obs_quantiles, mod_quantiles, label=name, color=info["color"], s=10, alpha=0.6)
                
        # Save CSV results
        df_out = pd.DataFrame(results)
        csv_filename = f"qqplot_insitu_{station}.csv"
        csv_path = os.path.join(csv_results_dir, csv_filename)
        df_out.to_csv(csv_path, index=False)
        print(f"[SUCCESS] Saved QQ data for station {station} to: {csv_path}")
        
        # Add 1:1 line
        max_val = np.nanmax([np.nanmax(obs_quantiles)] + [np.nanmax(results.get(name, [0])) for name in models_data])
        ax.plot([threshold, max_val], [threshold, max_val], 'k--', alpha=0.5, label="1:1 Perfect Fit")
        
        ax.set_title(f"Station: {station}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Observations (mm/day)")
        ax.set_ylabel("Downscaled / Gridded Predictions (mm/day)")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Standardize Legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.02 + 0.03 * nrows), fontsize=10)
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, "qqplot_insitu.png")
    plt.savefig(plot_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] In-situ QQ-plot figure saved to: {plot_path}")

if __name__ == "__main__":
    main()
