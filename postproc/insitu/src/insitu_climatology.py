import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF

# Add postproc/src to sys.path
root_path = str(Path(__file__).resolve().parents[2])
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import utils
import insitu_utils

class TaylorDiagram(object):
    """
    Taylor diagram polar plotter.
    """
    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5)):
        self.refstd = refstd
        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, np.pi/2, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1
        )

        if fig is None:
            fig = plt.figure()
            
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard Deviation (mm/day)")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("bottom")

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Reference point
        l, = self.ax.plot([0], self.refstd, 'k*', ls='', ms=10, label=label)
        t = np.linspace(0, np.pi/2)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(l)
        return l
        
    def add_contours(self, levels=5, **kwargs):
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, np.pi/2))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))
        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

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
    results_dir = os.path.join(root_path, "results", config.get('experiment', 'postproc'), "insitu")
    os.makedirs(results_dir, exist_ok=True)

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
        print("[WARNING] Requested stations not found in data. Processing first 3 available.")
        stations_to_process = list(stations_meta.keys())[:3]

    # === 2. Load Model Datasets ===
    print("[INFO] Pre-loading models and predictions...")
    models_data = {}
    
    # Add reference dataset
    ref_ds = xr.open_dataset(os.path.join(root_path, ref_cfg['file_path']))
    models_data[ref_cfg['name'].upper()] = {
        "ds": ref_ds,
        "var_name": ref_cfg['variable_name'],
        "color": "black",
        "marker": "o"
    }

    # Add model configurations
    colors_palette = ["red", "darkblue", "green", "lightblue", "orange", "purple"]
    markers_palette = ["^", "D", "s", "v", "p", "h"]
    
    for idx, d in enumerate(datasets_cfg):
        fpath = os.path.join(root_path, d['file_path'])
        if os.path.exists(fpath):
            ds_mod = xr.open_dataset(fpath)
            models_data[d['name'].upper()] = {
                "ds": ds_mod,
                "var_name": d['variable_name'],
                "color": colors_palette[idx % len(colors_palette)],
                "marker": markers_palette[idx % len(markers_palette)]
            }
        else:
            print(f"[WARNING] Prediction file not found: {fpath}. Skipping.")

    months = np.arange(1, 13)

    # ==========================================================
    # A. ANNUAL MONTHLY CLIMATOLOGY FIGURE
    # ==========================================================
    print("[INFO] Generating Annual Monthly Climatology Plots...")
    fig_clim, axes_clim = plt.subplots(1, len(stations_to_process), figsize=(6 * len(stations_to_process), 5), dpi=300)
    if len(stations_to_process) == 1:
        axes_clim = [axes_clim]
        
    for i, station in enumerate(stations_to_process):
        ax = axes_clim[i]
        st_obs = obs_df[obs_df["Station"] == station].copy().sort_values("Date")
        obs_series = st_obs.set_index("Date")["Precipitation"]
        
        # Plot OBS
        obs_clim = st_obs.groupby(st_obs["Date"].dt.month)["Precipitation"].mean().reindex(months)
        ax.plot(months, obs_clim.values, label="OBS", color="black", linewidth=3, linestyle="-")
        
        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]
        
        # Plot Models
        for name, info in models_data.items():
            mod_da = insitu_utils.extract_nearest_gridpoint(info["ds"], lat, lon, info["var_name"])
            aligned = insitu_utils.align_series(obs_series, mod_da)
            
            if len(aligned) > 0:
                aligned["month"] = aligned.index.month
                sim_clim = aligned.groupby("month")["sim"].mean().reindex(months)
                ax.plot(months, sim_clim.values, label=name, color=info["color"], linewidth=1.5)
                
        ax.set_title(f"Climatology: {station}", fontsize=12, fontweight='bold')
        ax.set_xticks(months)
        ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_xlabel("Month")
        ax.set_ylabel("Precipitation (mm/day)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
    # Standardize Legend
    handles, labels = axes_clim[0].get_legend_handles_labels()
    fig_clim.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.05), fontsize=10)
    plt.tight_layout()
    clim_path = os.path.join(results_dir, "annual_cycle_insitu.png")
    plt.savefig(clim_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Climatology annual cycle plot saved to: {clim_path}")

    # ==========================================================
    # B. TAYLOR DIAGRAM FIGURE
    # ==========================================================
    print("[INFO] Generating Taylor Diagrams...")
    fig_taylor = plt.figure(figsize=(5 * len(stations_to_process), 5), dpi=300)
    
    for i, station in enumerate(stations_to_process):
        st_obs = obs_df[obs_df["Station"] == station].copy().sort_values("Date")
        obs_series = st_obs.set_index("Date")["Precipitation"]
        obs_monthly = obs_series.resample("M").mean()
        
        ref_std = obs_monthly.std()
        rect = 100 + (10 * len(stations_to_process)) + (i + 1)
        
        # Initialize Taylor Diagram Subplot
        dia = TaylorDiagram(ref_std, fig=fig_taylor, rect=rect, label='OBS', srange=(0, 2.5))
        
        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]
        
        for name, info in models_data.items():
            mod_da = insitu_utils.extract_nearest_gridpoint(info["ds"], lat, lon, info["var_name"])
            aligned = insitu_utils.align_series(obs_series, mod_da)
            
            if len(aligned) > 0:
                aligned_monthly = aligned.resample("M").mean()
                stddev = aligned_monthly["sim"].std()
                corrcoef = np.corrcoef(aligned_monthly["obs"], aligned_monthly["sim"])[0, 1]
                
                # Check for NaNs
                if not np.isnan(stddev) and not np.isnan(corrcoef):
                    dia.add_sample(stddev, corrcoef, 
                                   marker=info["marker"], ms=8, ls='', 
                                   mfc=info["color"], mec=info["color"], 
                                   label=name)
                                   
        # Add Root Mean Square Error contours
        contours = dia.add_contours(levels=5, colors='0.5', linestyles=':')
        plt.clabel(contours, inline=1, fontsize=8, fmt='%.1f')
        dia._ax.set_title(f"Taylor Diagram: {station}", pad=20, fontsize=10, fontweight='bold')
        
        if i == 0:
            fig_taylor.legend(dia.samplePoints, 
                              [p.get_label() for p in dia.samplePoints], 
                              numpoints=1, prop=dict(size=8), loc='upper right', bbox_to_anchor=(0.95, 0.95))
            
    plt.tight_layout()
    taylor_path = os.path.join(results_dir, "taylor_diagram_insitu.png")
    plt.savefig(taylor_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Taylor diagram plot saved to: {taylor_path}")

if __name__ == "__main__":
    main()
