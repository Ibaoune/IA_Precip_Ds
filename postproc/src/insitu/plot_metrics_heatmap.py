"""
Author: M. El Aabaribaoune (@um6p)
Description: Generates comprehensive station-wise performance heatmaps evaluating RMSE, temporal correlation, and wet-day frequency bias for the retained models (GLM, CNN, Unet-exp32, ViT) against in-situ station data.
             Output: postproc/results/insitu_comparison/figures/
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings

# ── Global matplotlib settings ────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
})
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
INSITU_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(INSITU_ROOT, "src"))

# Postproc src for utils
POSTPROC_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if POSTPROC_SRC not in sys.path:
    sys.path.insert(0, POSTPROC_SRC)

import utils
import utils_insitu

# ── Output directory ───────────────────────────────────────────────────────────
# Will be initialized in main() after config is loaded

# ── Model order & colors ───────────────────────────────────────────────────────
MODEL_ORDER  = ["MSWEP", "GLM", "CNN", "Unet", "ViT"]
MODEL_COLORS = {
    "MSWEP": "black",
    "GLM":   "#F4A261",
    "CNN":   "#457B9D",
    "Unet":  "#1D3557",
    "ViT":   "#E63946",
}

# ── Metric computation ─────────────────────────────────────────────────────────
def compute_metrics(obs, sim, threshold=1.0):
    valid = ~np.isnan(obs) & ~np.isnan(sim)
    obs, sim = obs[valid], sim[valid]
    if len(obs) == 0:
        return {k: np.nan for k in ["Bias", "RMSE", "Correlation", "WetDayFreqBias"]}

    bias = float(np.mean(sim - obs))
    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    corr = float(np.corrcoef(obs, sim)[0, 1]) if len(obs) > 1 else np.nan

    years = len(obs) / 365.25
    obs_wet_freq = np.sum(obs >= threshold) / years
    sim_wet_freq = np.sum(sim >= threshold) / years
    wet_freq_bias = float(sim_wet_freq - obs_wet_freq)

    return {"Bias": bias, "RMSE": rmse, "Correlation": corr, "WetDayFreqBias": wet_freq_bias}


# ── Single heatmap helper ──────────────────────────────────────────────────────
def plot_single_heatmap(pivot, title, cbar_label, cmap, vmin, vmax, center,
                        fmt, out_path, letter=""):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot, annot=True, fmt=fmt,
        cmap=cmap, vmin=vmin, vmax=vmax, center=center,
        ax=ax, linewidths=0.5, linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.85},
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)
    if letter:
        ax.text(-0.06, 1.06, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    default_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "config_evaluation.yaml"))
    parser.add_argument("--config", default=default_cfg)
    args, _ = parser.parse_known_args()

    config = utils.load_config(args.config)
    params     = config["parameters"]
    ref_cfg    = config["reference"]
    datasets_cfg = config["datasets"]
    obs_cfg    = config.get("observations", {})

    OUT_DIR = config.get("output", {}).get("dir", "")
    os.makedirs(OUT_DIR, exist_ok=True)

    stations_list = params.get("stations", [])

    # 1. Load observations
    print("[INFO] Loading in-situ observations...")
    obs_df = utils_insitu.load_insitu_observations(
        excel_path=obs_cfg.get("excel_path", utils_insitu.DEFAULT_EXCEL_PATH),
        cache_path=obs_cfg.get("cache_path",  utils_insitu.DEFAULT_CACHE_PATH),
        start_date=params["start_date"],
        end_date=params["end_date"],
    )
    stations_meta = utils_insitu.get_station_metadata(obs_df)
    stations = [s for s in stations_list if s in stations_meta]

    # 2. Load model datasets
    print("[INFO] Loading model NetCDF files...")
    models_data = {}

    # Reference (MSWEP)
    ref_ds = xr.open_dataset(ref_cfg["file_path"])
    models_data["MSWEP"] = {"ds": ref_ds, "var_name": ref_cfg["variable_name"]}

    for d in datasets_cfg:
        fp = d["file_path"]
        if os.path.exists(fp):
            models_data[d["name"]] = {"ds": xr.open_dataset(fp), "var_name": d["variable_name"]}
            print(f"       -> Loaded {d['name']}")
        else:
            print(f"[WARNING] Missing: {fp}")

    # 3. Compute metrics
    print("[INFO] Computing metrics for each station × model...")
    all_metrics = []
    for station in stations:
        lat = stations_meta[station]["lat"]
        lon = stations_meta[station]["lon"]
        st_obs  = obs_df[obs_df["Station"] == station].sort_values("Date")
        obs_ser = st_obs.set_index("Date")["Precipitation"]

        for name, info in models_data.items():
            mod_da  = utils_insitu.extract_nearest_gridpoint(info["ds"], lat, lon, info["var_name"], search_radius=2)
            aligned = utils_insitu.align_series(obs_ser, mod_da)
            if len(aligned) > 0:
                m = compute_metrics(aligned["obs"].values, aligned["sim"].values)
                m["Station"] = station.title()
                m["Model"]   = name
                all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)

    # Save full CSV
    # csv_path = os.path.join(OUT_DIR, "station_metrics_summary.csv")
    # metrics_df.to_csv(csv_path, index=False)
    # print(f"[INFO] Metrics saved to {csv_path}")

    # 4. Build pivots (station × model)
    models_present = [m for m in MODEL_ORDER if m in metrics_df["Model"].unique()]

    def pivot(metric):
        pv = metrics_df.pivot(index="Station", columns="Model", values=metric)
        return pv[[m for m in models_present if m in pv.columns]]

    rmse_pv = pivot("RMSE")
    corr_pv = pivot("Correlation")
    wet_pv  = pivot("WetDayFreqBias")

    # 5. Individual heatmaps
    # --- RMSE ---
    plot_single_heatmap(
        rmse_pv,
        title="Station-wise Mean Performance (RMSE)",
        cbar_label="RMSE (mm day⁻¹)",
        cmap="viridis",
        vmin=None, vmax=None, center=None,
        fmt=".1f",
        out_path=os.path.join(OUT_DIR, "heatmap_rmse.png"),
        letter="(a)",
    )

    # --- Correlation ---
    plot_single_heatmap(
        corr_pv,
        title="Station-wise Temporal Correlation",
        cbar_label="Correlation (r)",
        cmap="magma",
        vmin=0, vmax=1, center=None,
        fmt=".2f",
        out_path=os.path.join(OUT_DIR, "heatmap_correlation.png"),
        letter="(b)",
    )

    # --- Wet-day Frequency Bias ---
    vmax_wet = float(np.nanmax(np.abs(wet_pv.values)))
    plot_single_heatmap(
        wet_pv,
        title="Wet-day Frequency Bias (Model − Station)",
        cbar_label="Bias (days/year)",
        cmap="RdBu_r",
        vmin=-vmax_wet, vmax=vmax_wet, center=0,
        fmt=".1f",
        out_path=os.path.join(OUT_DIR, "heatmap_wetday_bias.png"),
        letter="(c)",
    )

    # 6. Combined 3-panel figure (1 row × 3 cols)
    print("[INFO] Generating combined 3-panel figure...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def draw(ax, pv, title, cbar_label, cmap, vmin, vmax, center, fmt, letter):
        sns.heatmap(
            pv, annot=True, fmt=fmt,
            cmap=cmap, vmin=vmin, vmax=vmax, center=center,
            ax=ax, linewidths=0.5, linecolor="white",
            cbar_kws={"label": cbar_label, "shrink": 0.85},
            annot_kws={"size": 10, "weight": "bold"},
        )
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', rotation=0)
        ax.text(-0.06, 1.06, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold")

    draw(axes[0], rmse_pv, "Station-wise RMSE", "RMSE (mm day⁻¹)",
         "viridis", None, None, None, ".1f", "(a)")
    draw(axes[1], corr_pv, "Temporal Correlation", "Correlation (r)",
         "magma", 0, 1, None, ".2f", "(b)")
    draw(axes[2], wet_pv,  "Wet-day Freq. Bias", "Bias (days/year)",
         "RdBu_r", -vmax_wet, vmax_wet, 0, ".1f", "(c)")

    plt.tight_layout()
    combined_path = os.path.join(OUT_DIR, "heatmaps_combined.png")
    fig.savefig(combined_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Combined figure saved: {combined_path}")


if __name__ == "__main__":
    main()
