# Author: M. El Aabaribaoune (@um6p)
"""
regional_summary.py
====================
Produces regional heatmaps and grouped barplots summarising four skill metrics
(RMSE, R95 event-number bias, CDD bias, Pearson correlation) for each of the
four Moroccan sub-regions (North, Northeast, East, South) and four models
(GLM, CNN, UNET, ViT) – all from the pre-computed NetCDF result files produced
by the main postproc pipeline.

Usage (from postproc root):
 python3 src/regional_summary.py configs/config_retained.yaml
 python3 src/regional_summary.py configs/config_retained.yaml --period Annual
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd
import regionmask
from pathlib import Path

# ── import project utils ────────────────────────────────────────────────────
ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, os.path.join(ROOT, "src"))
import utils

# ── Morocco regional bounding boxes [lat_min, lat_max, lon_min, lon_max] ────
REGIONS = {
 "North": (33.0, 36.0, -6.0, 0.0),
 "Northeast": (33.0, 36.0, -3.0, 0.0), # Rif / Oriental corridor
 "East": (30.0, 33.0, -3.0, 0.0),
 "South": (21.0, 30.0, -17.0, 0.0),
}

MODELS = ["GLM", "UNET", "ViT", "CNN"]
REGION_NAMES = list(REGIONS.keys())

# ── metric catalogue (label, nc_subpath_template, variable_name, kind) ──────
# kind = 'bias' → compute (model - ref) spatial mean
# kind = 'direct' → take spatial mean of the stored metric field
METRICS = {
 "RMSE": {
 "kind": "direct",
 "var": "rmse",
 "subpath": "rmse/results/{model}_pr_allmorr_calcul_land_strategy_mean_first_corr_per_year_{period}.nc",
 "label": "RMSE (mm/day)",
 "fmt": ".2f",
 "cmap": "YlOrRd",
 "center": None,
 },
 "R95 Freq Bias": {
 "kind": "bias",
 "var": "r95p",
 "subpath_model": "r95_nbEvents_freq/results/{model}_pr_allmorr_calcul_land_strategy_daily_first_corr_per_year_{period}.nc",
 "subpath_ref": "r95_nbEvents_freq/results/mswep_pr_allmorr_calcul_land_strategy_daily_first_corr_per_year_{period}.nc",
 "label": "R95 event-number bias (%)",
 "fmt": ".1f",
 "cmap": "RdBu_r",
 "center": 0.0,
 },
 "CDD Bias": {
 "kind": "bias",
 "var": "cdd",
 "subpath_model": "cdd/cdd_1mm/results/{model}_pr_allmorr_calcul_land_strategy_per_year_corr_per_year_{period}.nc",
 "subpath_ref": "cdd/cdd_1mm/results/mswep_pr_allmorr_calcul_land_strategy_per_year_corr_per_year_{period}.nc",
 "label": "CDD bias (days)",
 "fmt": ".1f",
 "cmap": "RdBu_r",
 "center": 0.0,
 },
 "Correlation": {
 "kind": "direct",
 "var": "corr",
 "subpath": "correlation/results/{model}_pr_allmorr_calcul_land_strategy_per_year_corr_per_year_{period}.nc",
 "label": "Pearson Correlation",
 "fmt": ".2f",
 "cmap": "RdYlGn",
 "center": None,
 },
}


# ════════════════════════════════════════════════════════════════════════════
# Helper functions
# ════════════════════════════════════════════════════════════════════════════

def regional_mean(da: xr.DataArray, region: tuple, reg_name: str = None) -> float:
 """Average a 2-D (or 3-D with year) DataArray over a shapefile or lat/lon bounding box."""
 if reg_name:
 shp_map = {
 "North": "north.shp",
 "Northeast": "north_east.shp",
 "East": "east.shp",
 "South": "south.shp"
 }
 shp_file = shp_map.get(reg_name)
 if shp_file:
 shp_path = os.path.join(ROOT, "shape_files", shp_file)
 if os.path.exists(shp_path):
 try:
 # Load regional shapefile
 gdf = gpd.read_file(shp_path).to_crs("EPSG:4326").dissolve()
 gdf["name"] = [reg_name]
 gdf = gdf.reset_index(drop=True)
 
 # Create regionmask mask
 mask = regionmask.from_geopandas(gdf, names="name", name=reg_name).mask(da)
 
 # Mask and calculate mean
 da_masked = da.where(~mask.isnull())
 val = float(np.nanmean(da_masked.values))
 if not np.isnan(val):
 return val
 except Exception as e:
 print(f" [WARNING] Shapefile masking failed for {reg_name} ({e}). Falling back to bounding box.")

 # Fallback to simple bounding box
 lat_min, lat_max, lon_min, lon_max = region
 lat_name = "lat" if "lat" in da.dims else "latitude"
 lon_name = "lon" if "lon" in da.dims else "longitude"

 sel = da.sel(
 {lat_name: slice(lat_max, lat_min), # lat often stored descending
 lon_name: slice(lon_min, lon_max)}
 )
 # If still empty try ascending slice
 if sel.sizes[lat_name] == 0:
 sel = da.sel(
 {lat_name: slice(lat_min, lat_max),
 lon_name: slice(lon_min, lon_max)}
 )
 val = float(np.nanmean(sel.values))
 return val


def load_metric(base_dir, meta, model, period):
 """Load the model (and optional reference) NetCDF and return a DataArray."""
 try:
 if meta["kind"] == "direct":
 path = os.path.join(base_dir, meta["subpath"].format(model=model, period=period))
 ds = xr.open_dataset(path)
 da = ds[meta["var"]]
 # Collapse year dimension by mean if present
 if "year" in da.dims:
 da = da.mean("year")
 return da

 elif meta["kind"] == "bias":
 path_m = os.path.join(base_dir, meta["subpath_model"].format(model=model, period=period))
 path_r = os.path.join(base_dir, meta["subpath_ref"].format(period=period))
 ds_m = xr.open_dataset(path_m)
 ds_r = xr.open_dataset(path_r)
 da_m = ds_m[meta["var"]]
 da_r = ds_r[meta["var"]]
 if "year" in da_m.dims:
 da_m = da_m.mean("year")
 if "year" in da_r.dims:
 da_r = da_r.mean("year")
 # For R95 freq bias express as relative (%) if r95p is absolute count
 bias = da_m - da_r
 if "R95" in list(METRICS.keys())[list(METRICS.values()).index(meta)]:
 # relative bias %
 bias = (da_m - da_r) / (da_r + 1e-6) * 100
 return bias

 except FileNotFoundError as e:
 print(f" [WARNING] File not found: {e}")
 return None


def build_table(base_dir: str, period: str) -> dict[str, pd.DataFrame]:
 """Return one DataFrame per metric: rows=MODELS, cols=REGIONS, values=scalar."""
 tables = {}
 for metric_name, meta in METRICS.items():
 rows = {}
 for model in MODELS:
 row = {}
 da = load_metric(base_dir, meta, model, period)
 for reg_name, bbox in REGIONS.items():
 if da is not None:
 row[reg_name] = regional_mean(da, bbox, reg_name)
 else:
 row[reg_name] = np.nan
 rows[model] = row
 tables[metric_name] = pd.DataFrame(rows).T # shape: (models, regions)
 return tables


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════

def plot_heatmap(tables: dict, period: str, out_path: str, model_colors: dict):
 """
 2-row × 2-col figure: one heatmap per metric.
 Rows = models, Cols = regions.
 """
 metrics = list(tables.keys())
 fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200)
 axes_flat = axes.flatten()

 for ax, metric_name in zip(axes_flat, metrics):
 df = tables[metric_name] # (n_models, n_regions)
 meta = METRICS[metric_name]

 cmap = meta["cmap"]
 center = meta.get("center")

 vmin = df.min().min()
 vmax = df.max().max()

 if center is not None:
 # Symmetric diverging scale
 abs_max = max(abs(vmin), abs(vmax))
 vmin, vmax = -abs_max, abs_max
 norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
 else:
 norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

 im = ax.imshow(df.values, aspect="auto", cmap=cmap, norm=norm)
 cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
 cbar.set_label(meta["label"], fontsize=9)

 ax.set_xticks(range(len(REGION_NAMES)))
 ax.set_xticklabels(REGION_NAMES, fontsize=10, rotation=30, ha="right")
 ax.set_yticks(range(len(MODELS)))
 ax.set_yticklabels(MODELS, fontsize=10)
 ax.set_title(f"{metric_name} – {period}", fontsize=12, fontweight="bold")

 # Annotate cells
 fmt = meta["fmt"]
 for i, model in enumerate(MODELS):
 for j, region in enumerate(REGION_NAMES):
 val = df.loc[model, region]
 txt_color = "white" if (norm(val) < 0.25 or norm(val) > 0.75) else "black"
 ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
 fontsize=9, color=txt_color, fontweight="bold")

 plt.suptitle(f"Regional Model Performance – {period}", fontsize=14, fontweight="bold", y=1.01)
 plt.tight_layout()
 os.makedirs(os.path.dirname(out_path), exist_ok=True)
 plt.savefig(out_path, dpi=200, bbox_inches="tight")
 plt.close()
 print(f"[SUCCESS] Heatmap saved → {out_path}")


def plot_barplots(tables: dict, period: str, out_path: str, model_colors: dict):
 """
 2-row × 2-col figure: one grouped barplot per metric.
 X-axis = Regions, grouped bars = Models.
 """
 metrics = list(tables.keys())
 n_regions = len(REGION_NAMES)
 x = np.arange(n_regions)
 bar_width = 0.18
 offsets = np.linspace(-(len(MODELS) - 1) / 2, (len(MODELS) - 1) / 2, len(MODELS)) * bar_width

 fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200)
 axes_flat = axes.flatten()

 for ax, metric_name in zip(axes_flat, metrics):
 df = tables[metric_name]
 meta = METRICS[metric_name]

 for k, model in enumerate(MODELS):
 vals = [df.loc[model, r] for r in REGION_NAMES]
 color = model_colors.get(model, model_colors.get(model.upper(), "#888888"))
 bars = ax.bar(x + offsets[k], vals, bar_width * 0.95,
 label=model, color=color, edgecolor="white", linewidth=0.5)

 if METRICS[metric_name]["center"] == 0.0:
 ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

 ax.set_xticks(x)
 ax.set_xticklabels(REGION_NAMES, fontsize=11)
 ax.set_ylabel(meta["label"], fontsize=10)
 ax.set_title(f"{metric_name} – {period}", fontsize=12, fontweight="bold")
 ax.legend(fontsize=9, framealpha=0.7)
 ax.grid(axis="y", linestyle="--", alpha=0.4)
 ax.spines["top"].set_visible(False)
 ax.spines["right"].set_visible(False)

 plt.suptitle(f"Regional Model Performance – {period}", fontsize=14, fontweight="bold", y=1.01)
 plt.tight_layout()
 os.makedirs(os.path.dirname(out_path), exist_ok=True)
 plt.savefig(out_path, dpi=200, bbox_inches="tight")
 plt.close()
 print(f"[SUCCESS] Barplot saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
 parser = argparse.ArgumentParser(description="Regional heatmaps & barplots from postproc results")
 parser.add_argument("config", help="Path to config_retained.yaml")
 parser.add_argument("--period", default="Annual",
 choices=["Annual", "DJF", "MAM", "JJA", "SON"],
 help="Seasonal period to plot (default: Annual)")
 parser.add_argument("--all-periods", action="store_true",
 help="Generate figures for all available periods")
 args = parser.parse_args()

 config = utils.load_config(args.config)

 global MODELS
 MODELS = [d["name"] for d in config.get("datasets", [])]
 if not MODELS:
 MODELS = ["GLM", "UNET", "ViT", "CNN"]

 experiment = config.get("experiment", "postproc")
 start = config["parameters"]["start_date"]
 end = config["parameters"]["end_date"]
 region_tag = config["parameters"].get("region", "allmorr")

 # Model colors from config (fallback to defaults)
 model_colors_cfg = config.get("visualisation", {}).get("model_colors", {})
 model_colors = {}
 for m in MODELS:
 if m in model_colors_cfg:
 model_colors[m] = model_colors_cfg[m]
 else:
 m_lower = m.lower()
 if "unet" in m_lower or "u-net" in m_lower:
 model_colors[m] = "#1D3557" if "unified" in m_lower else "#457B9D"
 elif "cnn" in m_lower:
 model_colors[m] = "#2A9D8F" if "unified" in m_lower else "#E76F51"
 elif "vit" in m_lower:
 model_colors[m] = "#E63946" if "unified" in m_lower else "#F4A261"
 elif "glm" in m_lower:
 model_colors[m] = "#264653" if "unified" in m_lower else "#2A9D8F"
 else:
 model_colors[m] = "#888888"

 # Base directory where per-metric results are stored
 base_dir = os.path.join(
 ROOT, "results", experiment,
 f"{start}_{end}", region_tag, "test"
 )
 if not os.path.isdir(base_dir):
 print(f"[ERROR] Results directory not found: {base_dir}")
 sys.exit(1)

 # Output directory
 out_dir = os.path.join(base_dir, "regional_summary")
 os.makedirs(out_dir, exist_ok=True)

 periods = (
 ["Annual", "DJF", "MAM", "JJA", "SON"] if args.all_periods
 else [args.period]
 )

 for period in periods:
 print(f"\n{'='*60}")
 print(f" Processing period: {period}")
 print(f"{'='*60}")

 tables = build_table(base_dir, period)

 # Print summary tables to console
 for metric_name, df in tables.items():
 print(f"\n── {metric_name} ──")
 print(df.round(3).to_string())

 # Save CSV summary
 combined = pd.concat(
 {k: v for k, v in tables.items()},
 axis=0
 )
 csv_path = os.path.join(out_dir, f"regional_metrics_{period}.csv")
 combined.to_csv(csv_path)
 print(f"\n[INFO] CSV summary saved → {csv_path}")

 # Heatmap
 heatmap_path = os.path.join(out_dir, f"regional_heatmap_{period}.png")
 plot_heatmap(tables, period, heatmap_path, model_colors)

 # Barplot
 barplot_path = os.path.join(out_dir, f"regional_barplot_{period}.png")
 plot_barplots(tables, period, barplot_path, model_colors)

 print("\n[DONE] All regional summary figures generated.")


if __name__ == "__main__":
 main()
