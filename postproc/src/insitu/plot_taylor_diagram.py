"""
Author: M. El Aabaribaoune (@um6p)
Description: Generates Taylor diagrams for each station to synthesize model performance based on correlation, RMSE, and standard deviation.
"""

import os
import sys
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF

# Add postproc/src to sys.path
root_path = str(Path(__file__).resolve().parents[2])
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import utils
import utils_insitu

class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
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
            
        if isinstance(rect, tuple) or isinstance(rect, list):
            ax = FA.FloatingSubplot(fig, *rect, grid_helper=ghelper)
        else:
            ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        ax.axis["top"].set_axis_direction("bottom")
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("bottom")

        ax.axis["bottom"].set_visible(False)

        self._ax = ax
        self.ax = ax.get_aux_axes(tr)

        # Add reference point and stddev contour
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
    # Default config points to config_evaluation.yaml in the insitu directory
    default_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "config_evaluation.yaml"))
    parser.add_argument("--config", default=default_config, help="Path to config file")
    args, unknown = parser.parse_known_args()

    # === Load Configuration ===
    config = utils.load_config(args.config)
    
    params = config['parameters']
    ref_cfg = config['reference']
    datasets_cfg = config['datasets']
    obs_cfg = config.get('observations', {})

    stations = params.get('stations', [])

    excel_file = obs_cfg.get('excel_path', utils_insitu.DEFAULT_EXCEL_PATH)
    cache_path = obs_cfg.get('cache_path', utils_insitu.DEFAULT_CACHE_PATH)

    print(f"[INFO] Loading observations...")
    df = utils_insitu.load_insitu_observations(
        excel_path=excel_file,
        cache_path=cache_path,
        start_date=params['start_date'],
        end_date=params['end_date']
    )
    df["month"] = df["Date"].dt.month

    # Set up results directories
    insitu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = config.get("output", {}).get("dir", "")
    # csv_results_dir = os.path.join(results_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

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

    # Common colors and markers mapping
    colors = {
        "mswep": "black",
        "glm": "green",
        "unet": "darkblue",
        "vit": "red",
        "cnn": "lightblue"
    }

    markers = {
        "mswep": "o",
        "glm": "s",
        "unet": "D",
        "vit": "^",
        "cnn": "v"
    }

    # Grid configuration (max 3 columns)
    num_stations = len(stations)
    ncols = min(3, num_stations)
    nrows = int(np.ceil(num_stations / 3))

    fig = plt.figure(figsize=(5 * ncols, 6 * nrows), dpi=300)

    for i, station in enumerate(stations):
        print(f"[INFO] Processing station: {station}")
        st = df[df["Station"] == station].copy()
        st = st.sort_values("Date")

        lat = st["Latitude"].iloc[0]
        lon = st["Longitude"].iloc[0]

        obs_series = st.set_index("Date")["Precipitation"]
        obs_monthly = obs_series.resample("M").mean()

        # Create Taylor diagram for this station
        rect = (nrows, ncols, i + 1)
        ref_std = obs_monthly.std()
        
        # Initialize diagram
        dia = TaylorDiagram(ref_std, fig=fig, rect=rect, label='OBS', srange=(0, 2.5))
        
        for name, ds in datasets.items():
            point = utils_insitu.extract_nearest_gridpoint(ds, lat, lon)
            
            sim = pd.Series(
                point.values,
                index=pd.to_datetime(point["time"].values)
            )
            sim = sim.reindex(obs_series.index)
            sim_monthly = sim.resample("M").mean()
            
            # Calculate metrics
            merged = pd.concat([obs_monthly, sim_monthly], axis=1).dropna()
            merged.columns = ["obs", "sim"]
            
            if len(merged) > 0:
                stddev = merged["sim"].std()
                corrcoef = np.corrcoef(merged["obs"], merged["sim"])[0, 1]
                
                color = colors.get(name.lower(), "orange")
                marker = markers.get(name.lower(), "p")
                
                dia.add_sample(stddev, corrcoef, 
                               marker=marker, ms=10, ls='', 
                               mfc=color, mec=color, 
                               label=name.upper())

        # Add RMS contours
        contours = dia.add_contours(levels=5, colors='0.5')
        plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

        # Add title
        dia._ax.set_title(station, pad=20, fontweight='bold')
        
        # Add legend to the first plot
        if i == 0:
            fig.legend(dia.samplePoints, 
                       [p.get_label() for p in dia.samplePoints], 
                       numpoints=1, prop=dict(size='small'), loc='upper right')

    plt.tight_layout()
    out_file = os.path.join(results_dir, "taylor_diagram_insitu.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved Taylor diagram to {out_file}")

if __name__ == "__main__":
    main()
