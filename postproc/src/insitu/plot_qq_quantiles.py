"""
Author: M. El Aabaribaoune (@um6p)
Description: Generates Quantile-Quantile (Q-Q) plots on a logarithmic scale to evaluate extreme precipitation distributions across models and stations.
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Add postproc/src to sys.path
root_path = str(Path(__file__).resolve().parents[3])
src_path = os.path.join(root_path, "postproc", "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import utils
import utils_insitu

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

    # Set up results directories
    insitu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = config.get("output", {}).get("dir", "")
    # csv_results_dir = os.path.join(results_dir, "results")
    # os.makedirs(csv_results_dir, exist_ok=True)

    # Threshold and quantile configuration
    threshold = config.get('metric', {}).get('threshold', 1.0)
    n_quantiles = config.get('metric', {}).get('n_quantiles', 1000)

    # === 1. Load Station Observations ===
    obs_df = utils_insitu.load_insitu_observations(
        excel_path=obs_cfg.get('excel_path', utils_insitu.DEFAULT_EXCEL_PATH),
        cache_path=obs_cfg.get('cache_path', utils_insitu.DEFAULT_CACHE_PATH),
        start_date=params['start_date'],
        end_date=params['end_date']
    )
    stations_meta = utils_insitu.get_station_metadata(obs_df)
    stations_list = params.get('stations', [])
    stations_to_process = [s for s in stations_list if s in stations_meta]
    
    if not stations_to_process:
        print("[WARNING] Requested stations not found. Processing first 3 available.")
        stations_to_process = list(stations_meta.keys())[:3]

    # === 2. Load Model Datasets ===
    print("[INFO] Pre-loading models and predictions...")

    # ── Canonical model display names & colors ──────────────────────────────
    MODEL_STYLE = {
        "MSWEP": {"label": "MSWEP", "color": "#000000", "marker": "o", "zorder": 3, "alpha": 0.70},
        "GLM":   {"label": "GLM",   "color": "#D55E00", "marker": "^", "zorder": 4, "alpha": 0.85},
        "CNN":   {"label": "CNN",   "color": "#0072B2", "marker": "s", "zorder": 5, "alpha": 0.85},
        "UNET":  {"label": "U-Net", "color": "#009E73", "marker": "D", "zorder": 6, "alpha": 0.85},
        "VIT":   {"label": "ViT",   "color": "#CC79A7", "marker": "p", "zorder": 7, "alpha": 0.85},
    }

    models_data = {}

    # Add reference dataset (e.g. MSWEP)
    ref_key = ref_cfg['name'].upper()
    ref_ds  = xr.open_dataset(os.path.join(root_path, ref_cfg['file_path']))
    ref_style = MODEL_STYLE.get(ref_key, {"label": ref_cfg['name'],
                                           "color": "#000000", "marker": "o", "zorder": 3, "alpha": 0.70})
    models_data[ref_key] = {
        "ds":       ref_ds,
        "var_name": ref_cfg['variable_name'],
        **ref_style,
    }

    # Add downscaled predictions
    for idx, d in enumerate(datasets_cfg):
        key = d['name'].upper()
        fpath = d['file_path'] if os.path.isabs(d['file_path']) \
                else os.path.join(root_path, d['file_path'])
        if os.path.exists(fpath):
            ds_mod = xr.open_dataset(fpath)
            style = MODEL_STYLE.get(key, {"label": d['name'], "color": "gray",
                                          "marker": "o", "zorder": 4, "alpha": 0.80})
            models_data[key] = {
                "ds": ds_mod,
                "var_name": d['variable_name'],
                **style,
            }
        else:
            print(f"[WARNING] Prediction file not found: {fpath}. Skipping.")

    # ── Figure layout: 2 rows × 4 cols (7 data panels + 1 legend panel) ─────
    import matplotlib.gridspec as gridspec
    import matplotlib.font_manager as fm
    import matplotlib.lines as mlines

    PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]
    STATION_NAMES = {
        "CASABLANCA":  "Casablanca",
        "TANGER":      "Tanger",
        "FES":         "Fes",
        "OUJDA":       "Oujda",
        "OUARZAZATE":  "Ouarzazate",
        "DAKHLA":      "Dakhla",
        "MARRAKECH":   "Marrakech",
    }

    # Font sizes
    FS_TITLE  = 10
    FS_TICK   = 8.5
    FS_LEGEND = 8.0
    FONT_FAM  = "DejaVu Sans"

    plt.rcParams.update({
        "font.family":       FONT_FAM,
        "font.size":         FS_TICK,
        "axes.linewidth":    0.6,
        "axes.edgecolor":    "#b0b0b0",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.color":       "#555555",
        "ytick.color":       "#555555",
    })

    FIG_W, FIG_H = 7.2, 3.5   # inches — fits a 2-column journal page
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=600)

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        left=0.04, right=0.98,
        bottom=0.08, top=0.93,
        hspace=0.28, wspace=0.25,
    )

    axes_data = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    # Last panel → legend only
    ax_legend = axes_data[7]
    ax_legend.axis("off")

    # ── Per-station scatter & CSV ─────────────────────────────────────────────
    for i, station in enumerate(stations_to_process[:7]):
        ax = axes_data[i]
        st_obs = obs_df[obs_df["Station"] == station].copy().sort_values("Date")
        obs_series = st_obs.set_index("Date")["Precipitation"]

        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]

        obs_quantiles = compute_qq_quantiles(obs_series,
                                             n_quantiles=n_quantiles,
                                             threshold=threshold)

        results = {
            "quantile_rank": np.linspace(0, 1, n_quantiles),
            "OBS": obs_quantiles,
        }

        all_vals = [obs_quantiles]
        for name, info in models_data.items():
            mod_da  = utils_insitu.extract_nearest_gridpoint(
                info["ds"], lat, lon, info["var_name"])
            aligned = utils_insitu.align_series(obs_series, mod_da)

            if len(aligned) > 0:
                mod_q = compute_qq_quantiles(aligned["sim"],
                                             n_quantiles=n_quantiles,
                                             threshold=threshold)
                results[name] = mod_q
                all_vals.append(mod_q)

                kwargs = {
                    "label": info["label"],
                    "s": 6 if name == "MSWEP" else 8,
                    "marker": info.get("marker", "o"),
                    "alpha": info["alpha"],
                    "zorder": info["zorder"],
                    "color": info["color"],
                    "linewidths": 0
                }

                ax.scatter(obs_quantiles, mod_q, **kwargs)

        # Save CSV
        df_out = pd.DataFrame(results)
        # csv_path = os.path.join(csv_results_dir, f"qqplot_insitu_{station}.csv")
        # df_out.to_csv(csv_path, index=False)
        # print(f"[SUCCESS] Saved QQ data for station {station} to: {csv_path}")

        # Exact axis limits for all stations
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 80)
        ax.set_xticks([0, 25, 50, 75])
        ax.set_yticks([0, 20, 40, 60, 80])

        # 1:1 line (only valid common range)
        ax.plot([0, 75], [0, 75],
                color="#999999", linestyle="--", linewidth=0.8,
                alpha=0.6, zorder=2)

        # Grid (lighter than 1:1 line)
        ax.grid(True, linestyle="--", linewidth=0.4,
                color="#e0e0e0", alpha=0.8, zorder=1)
        ax.set_axisbelow(True)

        # Panel title: bold letter + station name
        label        = PANEL_LABELS[i]
        name_display = STATION_NAMES.get(station, station.capitalize())
        
        # Use MathText for bold panel letter
        ax.set_title(r"$\mathbf{" + label + r"}$ " + name_display,
                     loc="left", fontsize=FS_TITLE, pad=1)

        # Tick font size
        ax.tick_params(axis="both", labelsize=FS_TICK)

    # ── Legend in 8th panel ───────────────────────────────────────────────────
    legend_handles = []
    # Models in display order
    for key in ["MSWEP", "GLM", "CNN", "UNET", "VIT"]:
        if key in models_data:
            s = models_data[key]
            marker_kwargs = {
                "marker": s.get("marker", "o"),
                "linestyle": "None",
                "markersize": 3.5 if key == "MSWEP" else 4.5,
                "label": s["label"],
                "alpha": s["alpha"],
                "color": s["color"],
                "markeredgewidth": 0
            }
                
            legend_handles.append(mlines.Line2D([], [], **marker_kwargs))
    # 1:1 line entry
    legend_handles.append(
        mlines.Line2D(
            [], [],
            color="#999999", linestyle="--",
            linewidth=1.2, label="1:1 Perfect Fit",
        )
    )
    ax_legend.legend(
        handles=legend_handles,
        loc="center",
        fontsize=FS_LEGEND,
        frameon=True,
        framealpha=0.8,
        edgecolor="none",
        ncol=1,
        handlelength=1.5,
        handletextpad=0.6,
        borderpad=0.4,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    plot_path_png = os.path.join(results_dir, "qqplot_insitu.png")
    # plot_path_pdf = os.path.join(results_dir, "qqplot_insitu.pdf")
    
    # Save to disk
    plt.savefig(plot_path_png, dpi=600, bbox_inches="tight")
    # plt.savefig(plot_path_pdf, bbox_inches="tight")
    plt.close()
    print(f"\n[SUCCESS] Main plot saved to: {plot_path_png}")

if __name__ == "__main__":
    main()
