"""
Author: M. El Aabaribaoune (@um6p)
Description: Produces a unified diagnostic map illustrating the distinct grid points selected by each model for every station, including fallback handling for missing data.
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
root_path = str(Path(__file__).resolve().parents[3])
src_path  = os.path.join(root_path, "postproc", "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import utils
import utils_insitu

warnings.filterwarnings("ignore")

# ── Helper: find actual grid point used (mirrors utils_insitu logic) ─────────
def find_used_gridpoint(ds, lat, lon, var_name, search_radius=2):
    lat_key = "lat" if "lat" in ds.coords else "latitude"
    lon_key = "lon" if "lon" in ds.coords else "longitude"
    da = ds[var_name]

    # --- TANGER OVERRIDE ---
    if abs(lat - 35.72) < 0.01 and abs(lon - -5.90) < 0.01:
        lat = 35.75
        lon = -5.85
        
    # --- DAKHLA OVERRIDE ---
    if abs(lat - 23.72) < 0.01 and abs(lon - -15.93) < 0.01:
        lat = 23.75
        lon = -15.75
    # -----------------------

    idx_lat = int(np.abs(ds[lat_key].values - lat).argmin())
    idx_lon = int(np.abs(ds[lon_key].values - lon).argmin())

    point = da.isel({lat_key: idx_lat, lon_key: idx_lon})
    used_lat = float(ds[lat_key].values[idx_lat])
    used_lon = float(ds[lon_key].values[idx_lon])
    used_i, used_j = idx_lat, idx_lon

    if search_radius > 0 and point.isnull().all().values:
        min_dist = float("inf")
        for i in range(max(0, idx_lat - search_radius),
                       min(len(ds[lat_key]), idx_lat + search_radius + 1)):
            for j in range(max(0, idx_lon - search_radius),
                           min(len(ds[lon_key]), idx_lon + search_radius + 1)):
                if i == idx_lat and j == idx_lon:
                    continue
                cand = da.isel({lat_key: i, lon_key: j})
                if not cand.isnull().all().values:
                    c_lat = float(ds[lat_key].values[i])
                    c_lon = float(ds[lon_key].values[j])
                    dist = (c_lat - lat)**2 + (c_lon - lon)**2
                    if dist < min_dist:
                        min_dist = dist
                        used_lat, used_lon = c_lat, c_lon
                        used_i, used_j = i, j

    nearest_is_nan = da.isel({lat_key: idx_lat, lon_key: idx_lon}).isnull().all().values
    return {
        "used_lat": used_lat,
        "used_lon": used_lon,
        "nearest_lat": float(ds[lat_key].values[idx_lat]),
        "nearest_lon": float(ds[lon_key].values[idx_lon]),
        "nearest_was_nan": bool(nearest_is_nan),
        "fallback_used": bool(nearest_is_nan and (used_i != idx_lat or used_j != idx_lon)),
    }


def main():
    import argparse
    default_cfg = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config_evaluation.yaml")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_cfg)
    args, _ = parser.parse_known_args()

    config       = utils.load_config(args.config)
    params       = config["parameters"]
    ref_cfg      = config["reference"]
    datasets_cfg = config["datasets"]
    obs_cfg      = config.get("observations", {})

    insitu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = config.get("output", {}).get("dir", "")
    os.makedirs(out_dir, exist_ok=True)

    # ── Observations ──────────────────────────────────────────────────────────
    obs_df = utils_insitu.load_insitu_observations(
        excel_path=obs_cfg.get("excel_path", utils_insitu.DEFAULT_EXCEL_PATH),
        cache_path=obs_cfg.get("cache_path", utils_insitu.DEFAULT_CACHE_PATH),
        start_date=params["start_date"],
        end_date=params["end_date"],
    )
    stations_meta = utils_insitu.get_station_metadata(obs_df)
    stations_list = params.get("stations", [])
    stations_to_process = [s for s in stations_list if s in stations_meta]

    # ── Models ────────────────────────────────────────────────────────────────
    root_data = str(Path(__file__).resolve().parents[3])

    models_info = {}
    # reference
    ref_path = os.path.join(root_data, ref_cfg["file_path"]) if not os.path.isabs(ref_cfg["file_path"]) else ref_cfg["file_path"]
    if os.path.exists(ref_path):
        models_info[ref_cfg["name"].upper()] = {
            "path": ref_path, "var": ref_cfg["variable_name"]
        }

    for d in datasets_cfg:
        fpath = d["file_path"] if os.path.isabs(d["file_path"]) else os.path.join(root_data, d["file_path"])
        if os.path.exists(fpath):
            models_info[d["name"].upper()] = {"path": fpath, "var": d["variable_name"]}
        else:
            print(f"[WARNING] Not found: {fpath}")

    # ── Open datasets (lazy) ──────────────────────────────────────────────────
    datasets_open = {}
    for name, info in models_info.items():
        print(f"[INFO] Opening {name} ...")
        datasets_open[name] = xr.open_dataset(info["path"])

    # ── Build coordinate table ────────────────────────────────────────────────
    rows = []
    gp_store = {}   # station → model → {used_lat, used_lon, fallback_used}

    for station in stations_to_process:
        st_lat = stations_meta[station]["lat"]
        st_lon = stations_meta[station]["lon"]
        gp_store[station] = {}

        for model_name, ds in datasets_open.items():
            var_name = models_info[model_name]["var"]
            info_gp = find_used_gridpoint(ds, st_lat, st_lon, var_name)
            gp_store[station][model_name] = info_gp

            rows.append({
                "Station":       station,
                "Sta_lat":       round(st_lat, 4),
                "Sta_lon":       round(st_lon, 4),
                "Model":         model_name,
                "Grid_lat":      round(info_gp["used_lat"], 4),
                "Grid_lon":      round(info_gp["used_lon"], 4),
                "Nearest_lat":   round(info_gp["nearest_lat"], 4),
                "Nearest_lon":   round(info_gp["nearest_lon"], 4),
                "Nearest_NaN":   info_gp["nearest_was_nan"],
                "Fallback_used": info_gp["fallback_used"],
                "Δlat":          round(info_gp["used_lat"] - st_lat, 4),
                "Δlon":          round(info_gp["used_lon"] - st_lon, 4),
                "Distance_deg":  round(np.sqrt((info_gp["used_lat"]-st_lat)**2 +
                                               (info_gp["used_lon"]-st_lon)**2), 4),
            })

    df_table = pd.DataFrame(rows)
    # txt_path = os.path.join(out_dir, "gridpoints_table.txt")
    # with open(txt_path, "w") as f:
    #     f.write(df_table.to_string(index=False))
    # csv_path = os.path.join(out_dir, "gridpoints_table.csv")
    # df_table.to_csv(csv_path, index=False)
    # print(f"[SUCCESS] Coordinate table saved to: {txt_path}")
    print(df_table.to_string(index=False))

    # ── Map ──────────────────────────────────────────────────────────────────
    model_colors = {
        "MSWEP": "black",
        "GLM":   "red",
        "CNN":   "darkblue",
        "UNET":  "green",
        "VIT":   "skyblue",
    }
    # Fallback colors for unexpected model names
    extra_colors = ["orange", "purple", "brown", "pink"]
    ci = 0
    for name in datasets_open:
        if name not in model_colors:
            model_colors[name] = extra_colors[ci % len(extra_colors)]
            ci += 1

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([-18, 2, 20, 38], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="lightcyan", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":", zorder=1)
    ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")

    # Plot stations (large stars)
    for station in stations_to_process:
        st_lat = stations_meta[station]["lat"]
        st_lon = stations_meta[station]["lon"]
        ax.plot(st_lon, st_lat, marker="*", markersize=14,
                color="gold", markeredgecolor="black", markeredgewidth=0.8,
                transform=ccrs.PlateCarree(), zorder=5)
        ax.text(st_lon + 0.2, st_lat + 0.15, station, fontsize=7, fontweight="bold",
                transform=ccrs.PlateCarree(), zorder=6)

    # Plot grid points per model
    n_models = len(datasets_open)
    offsets   = np.linspace(-0.1, 0.1, n_models)  # small horizontal jitter

    for ki, (model_name, ds) in enumerate(datasets_open.items()):
        color = model_colors.get(model_name, "gray")
        for station in stations_to_process:
            st_lat = stations_meta[station]["lat"]
            st_lon = stations_meta[station]["lon"]
            gp = gp_store[station][model_name]

            # Draw line from station → used grid point
            ax.plot([st_lon, gp["used_lon"]], [st_lat, gp["used_lat"]],
                    color=color, linewidth=0.8, alpha=0.6,
                    transform=ccrs.PlateCarree(), zorder=3)

            marker = "^" if gp["fallback_used"] else "o"
            ax.plot(gp["used_lon"] + offsets[ki], gp["used_lat"],
                    marker=marker, markersize=7, color=color,
                    markeredgecolor="white", markeredgewidth=0.5,
                    transform=ccrs.PlateCarree(), zorder=4,
                    label=model_name if station == stations_to_process[0] else "")

    # Legend for models
    model_patches = [
        mpatches.Patch(color=model_colors.get(m, "gray"), label=m)
        for m in datasets_open
    ]
    star_patch = plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
                            markeredgecolor="black", markersize=12, label="Station")
    circle_patch = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                              markersize=8, label="Nearest grid pt (direct)")
    tri_patch    = plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                              markersize=8, label="Fallback grid pt (NaN → neighbor)")

    ax.legend(handles=model_patches + [star_patch, circle_patch, tri_patch],
              loc="lower left", fontsize=8, framealpha=0.85)

    ax.set_title("Station locations & nearest grid points used per model\n"
                 "(▲ = fallback point used because nearest was NaN)",
                 fontsize=11, fontweight="bold")

    map_path = os.path.join(out_dir, "gridpoints_map.png")
    plt.savefig(map_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Map saved to: {map_path}")


if __name__ == "__main__":
    main()
