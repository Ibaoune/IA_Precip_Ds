"""
Author: M. El Aabaribaoune (@um6p)
Description: Part of the post-processing and evaluation pipeline for the downscaling project.
"""

import os
import sys
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add postproc/src to sys.path
root_path = str(Path(__file__).resolve().parents[2])
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import utils
import insitu_utils

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

    stations = params.get('stations', ["CASABLANCA", "FES", "BGE TANGER MED", "DAKHLA", "AGADIR"])
    months = np.arange(1, 13)

    excel_file = obs_cfg.get('excel_path', insitu_utils.DEFAULT_EXCEL_PATH)
    cache_path = obs_cfg.get('cache_path', insitu_utils.DEFAULT_CACHE_PATH)

    # Set up results directories
    insitu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(insitu_root, "Figs")
    os.makedirs(results_dir, exist_ok=True)

    # ==========================================================
    # LOAD OBS
    # ==========================================================
    print(f"[INFO] Loading observations...")
    df = insitu_utils.load_insitu_observations(
        excel_path=excel_file,
        cache_path=cache_path,
        start_date=params['start_date'],
        end_date=params['end_date']
    )
    df["month"] = df["Date"].dt.month

    # ==========================================================
    # LOAD MODELS
    # ==========================================================
    print("[INFO] Loading NetCDF datasets...")
    datasets = {}
    
    # Add reference dataset (e.g. MSWEP)
    ref_ds = xr.open_dataset(os.path.join(root_path, ref_cfg['file_path']))
    ref_ds["time"] = pd.to_datetime(ref_ds["time"].values)
    datasets[ref_cfg['name'].lower()] = ref_ds

    # Add downscaled predictions
    for d in datasets_cfg:
        fpath = os.path.join(root_path, d['file_path'])
        if os.path.exists(fpath):
            print(f"       -> Loading {d['name']}...")
            ds = xr.open_dataset(fpath)
            ds["time"] = pd.to_datetime(ds["time"].values)
            datasets[d['name'].lower()] = ds
        else:
            print(f"[WARNING] Prediction file not found: {fpath}. Skipping.")

    # Colors mapping
    colors = {
        "mswep": "black",
        "glm": "green",
        "unet": "darkblue",
        "vit": "red",
        "cnn": "lightblue"
    }

    # ==========================================================
    # FIGURE GRID SETUP (max 3 stations per line)
    # ==========================================================
    print("[INFO] Initializing figure...")
    num_stations = len(stations)
    ncols = min(3, num_stations)
    nrows = int(np.ceil(num_stations / 3))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    
    # Flatten and wrap axes list
    if nrows == 1 and ncols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    # Hide any unused axes in the grid
    for idx in range(num_stations, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # ==========================================================
    # LOOP STATIONS
    # ==========================================================
    for i, station in enumerate(stations):
        print(f"[INFO] Processing station {i+1}/{len(stations)}: {station}")
        ax = axes_flat[i]

        st = df[df["Station"] == station].copy()
        st = st.sort_values("Date")

        lat = st["Latitude"].iloc[0]
        lon = st["Longitude"].iloc[0]

        # ======================================================
        # OBS CLIMATOLOGY
        # ======================================================
        obs_clim = st.groupby("month")["Precipitation"].mean().reindex(months)
        ax.plot(months, obs_clim.values, label="OBS", color="black", linewidth=3)

        metric_text = []
        obs_series = st.set_index("Date")["Precipitation"]

        # ======================================================
        # MODELS
        # ======================================================
        for name, ds in datasets.items():
            point = insitu_utils.extract_nearest_gridpoint(ds, lat, lon)
            sim = pd.Series(
                point.values,
                index=pd.to_datetime(point["time"].values)
            )
            sim = sim.reindex(obs_series.index)

            # Climatology
            sim_df = pd.DataFrame({
                "value": sim.values,
                "month": sim.index.month
            })
            sim_clim = sim_df.groupby("month")["value"].mean().reindex(months)

            # Plot if valid
            if not np.all(np.isnan(sim_clim.values)):
                color = colors.get(name.lower(), "orange")
                ax.plot(months, sim_clim.values, label=name.upper(), color=color)

            # Metrics
            merged = pd.concat([obs_series, sim], axis=1)
            merged.columns = ["obs", "sim"]
            merged = merged.dropna()

            if len(merged) > 0:
                rmse = np.sqrt(np.mean((merged["obs"] - merged["sim"])**2))
                bias = np.mean(merged["sim"] - merged["obs"])
                metric_text.append(f"{name.upper()}: RMSE={rmse:.2f} | Bias={bias:.2f}")

        # ======================================================
        # STYLE
        # ======================================================
        ax.set_title(station, fontsize=12, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Precipitation (mm/day)")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(labels),
                fontsize=10,
                bbox_to_anchor=(0.5, 1.02 + 0.03 * nrows)
            )

        # Metrics box
        ax.text(
            0.02, 0.95,
            "\n".join(metric_text),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7)
        )

    # ==========================================================
    # SAVE
    # ==========================================================
    out_path = os.path.join(results_dir, "annual_cycle_FIXED.png")
    print(f"[INFO] Saving plot to {out_path}...")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[INFO] Done!")

if __name__ == "__main__":
    main()
