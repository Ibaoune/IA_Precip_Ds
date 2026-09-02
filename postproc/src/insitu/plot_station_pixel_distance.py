"""
Author: M. El Aabaribaoune (@um6p)
Description: Generates a spatial diagnostic map to verify the exact distance between physical station coordinates and the extracted model grid pixels.
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
from pathlib import Path

# Prevent cartopy download warnings
warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
root_path = str(Path(__file__).resolve().parents[3])
src_path  = os.path.join(root_path, "postproc", "src")
if src_path not in sys.path:
    sys.path.append(src_path)

import utils
import utils_insitu

def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    R = 6371.0 # Earth radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def find_used_gridpoint(ds, lat, lon, var_name, search_radius=2):
    """Mirrors exact extraction logic from utils_insitu.py."""
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
    is_valid = not da.isel({lat_key: used_i, lon_key: used_j}).isnull().all().values
    
    return {
        "used_lat": used_lat,
        "used_lon": used_lon,
        "used_i": used_i,
        "used_j": used_j,
        "nearest_lat": float(ds[lat_key].values[idx_lat]),
        "nearest_lon": float(ds[lon_key].values[idx_lon]),
        "nearest_was_nan": bool(nearest_is_nan),
        "fallback_used": bool(nearest_is_nan and (used_i != idx_lat or used_j != idx_lon)),
        "is_valid": bool(is_valid)
    }

def main():
    import argparse
    default_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "config_evaluation.yaml"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_cfg)
    args, _ = parser.parse_known_args()

    config       = utils.load_config(args.config)
    params       = config["parameters"]
    ref_cfg      = config["reference"]
    obs_cfg      = config.get("observations", {})

    stations_list = params.get('stations', [])
    PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    insitu_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = config.get("output", {}).get("dir", "")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load Observations ─────────────────────────────────────────────────────
    obs_df = utils_insitu.load_insitu_observations(
        excel_path=obs_cfg.get("excel_path", utils_insitu.DEFAULT_EXCEL_PATH),
        cache_path=obs_cfg.get("cache_path", utils_insitu.DEFAULT_CACHE_PATH),
        start_date=params["start_date"],
        end_date=params["end_date"],
    )
    stations_meta = utils_insitu.get_station_metadata(obs_df)
    stations_to_process = [s for s in stations_list if s in stations_meta]

    if len(stations_to_process) != 7:
        print(f"[WARNING] Expected 7 stations, got {len(stations_to_process)}")

    # ── Load Gridded Dataset (Reference: MSWEP) ──────────────────────────────
    ref_path = os.path.join(root_path, ref_cfg["file_path"]) if not os.path.isabs(ref_cfg["file_path"]) else ref_cfg["file_path"]
    print(f"[INFO] Opening reference dataset: {ref_path}")
    ds = xr.open_dataset(ref_path)
    var_name = ref_cfg["variable_name"]
    
    # Calculate grid resolution
    lat_key = "lat" if "lat" in ds.coords else "latitude"
    lon_key = "lon" if "lon" in ds.coords else "longitude"
    res_lat = abs(float(ds[lat_key].values[1] - ds[lat_key].values[0]))
    res_lon = abs(float(ds[lon_key].values[1] - ds[lon_key].values[0]))
    diag_deg = np.sqrt(res_lat**2 + res_lon**2)
    diag_km = diag_deg * 111.32  # Rough approximation for thresholds
    
    print(f"[INFO] Grid resolution: {res_lat:.3f} deg x {res_lon:.3f} deg. Diagonal ~ {diag_km:.2f} km")

    # ── Process Stations ──────────────────────────────────────────────────────
    rows = []
    
    for station in stations_to_process:
        st_lat = stations_meta[station]["lat"]
        st_lon = stations_meta[station]["lon"]

        # Ensure correct longitude conventions (dataset is MSWEP, often [-180, 180])
        # The extraction script uses raw coordinates without modification, we do the same.
        
        info = find_used_gridpoint(ds, st_lat, st_lon, var_name)
        dist_km = haversine(st_lon, st_lat, info["used_lon"], info["used_lat"])
        
        # Distance flag
        if dist_km <= 0.5 * diag_km:
            flag = "OK"
        elif dist_km <= 1.0 * diag_km:
            flag = "CHECK"
        else:
            flag = "FAR"
            
        rows.append({
            "station_name": station,
            "station_latitude": st_lat,
            "station_longitude": st_lon,
            "pixel_latitude": info["used_lat"],
            "pixel_longitude": info["used_lon"],
            "grid_row": info["used_i"],
            "grid_column": info["used_j"],
            "distance_km": dist_km,
            "grid_name": ref_cfg["name"],
            "pixel_valid": info["is_valid"],
            "fallback_used": info["fallback_used"],
            "distance_flag": flag
        })
        
        # Tanger specific print
        if station == "TANGER":
            print("\n==================================================")
            print("TANGER DIAGNOSTIC REPORT")
            print("==================================================")
            print(f"Station coordinates:     {st_lat:.6f}, {st_lon:.6f}")
            print(f"Selected pixel coords:   {info['used_lat']:.6f}, {info['used_lon']:.6f}")
            print(f"Grid indices (row, col): {info['used_i']}, {info['used_j']}")
            print(f"Distance in kilometres:  {dist_km:.4f} km")
            print(f"Grid resolution:         {res_lat:.3f} x {res_lon:.3f} degrees")
            print(f"Pixel is valid data:     {'Yes' if info['is_valid'] else 'No (NaN)'}")
            print(f"Fallback neighbor used:  {'Yes' if info['fallback_used'] else 'No'}")
            print(f"Distance flag:           {flag}")
            print("==================================================\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    # csv_path = os.path.join(out_dir, "nearest_station_pixel_coordinates.csv")
    # df_out.to_csv(csv_path, index=False)
    # print(f"[SUCCESS] Coordinates saved to: {csv_path}")

    # ── Main 2x4 Figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), dpi=300)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3, left=0.03, right=0.97, bottom=0.05, top=0.95)
    
    # Pre-compute local map extent: +/- 0.4 degrees for all
    margin = 0.4

    for i, station in enumerate(stations_to_process):
        ax = fig.add_subplot(gs[i // 4, i % 4], projection=ccrs.PlateCarree())
        
        row = df_out[df_out["station_name"] == station].iloc[0]
        st_lat, st_lon = row["station_latitude"], row["station_longitude"]
        px_lat, px_lon = row["pixel_latitude"], row["pixel_longitude"]
        
        ax.set_extent([st_lon - margin, st_lon + margin, st_lat - margin, st_lat + margin], crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f3f7", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=1.0, color="#333333", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle=":", zorder=1)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        # Determine nearby grid cells to display centers
        lat_vals = ds[lat_key].values
        lon_vals = ds[lon_key].values
        
        idx_lat_min = np.abs(lat_vals - (st_lat - margin)).argmin()
        idx_lat_max = np.abs(lat_vals - (st_lat + margin)).argmin()
        idx_lon_min = np.abs(lon_vals - (st_lon - margin)).argmin()
        idx_lon_max = np.abs(lon_vals - (st_lon + margin)).argmin()
        
        i_min, i_max = min(idx_lat_min, idx_lat_max), max(idx_lat_min, idx_lat_max)
        j_min, j_max = min(idx_lon_min, idx_lon_max), max(idx_lon_min, idx_lon_max)
        
        # Plot grid centers
        for ii in range(i_min, i_max + 1):
            for jj in range(j_min, j_max + 1):
                ax.plot(lon_vals[jj], lat_vals[ii], marker='+', color='#999999', markersize=4, transform=ccrs.PlateCarree(), zorder=2)
                
        # Highlight selected cell boundaries (assuming regular grid)
        cell_lat = [px_lat - res_lat/2, px_lat + res_lat/2, px_lat + res_lat/2, px_lat - res_lat/2, px_lat - res_lat/2]
        cell_lon = [px_lon - res_lon/2, px_lon - res_lon/2, px_lon + res_lon/2, px_lon + res_lon/2, px_lon - res_lon/2]
        ax.plot(cell_lon, cell_lat, color='orange', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=3)
        ax.fill(cell_lon, cell_lat, color='orange', alpha=0.2, transform=ccrs.PlateCarree(), zorder=3)

        # Plot connection line
        ax.plot([st_lon, px_lon], [st_lat, px_lat], color='red', linestyle='--', linewidth=1.2, transform=ccrs.PlateCarree(), zorder=4)

        # Plot selected pixel center
        ax.plot(px_lon, px_lat, marker='o', color='orangered', markersize=6, transform=ccrs.PlateCarree(), zorder=5)

        # Plot station
        ax.plot(st_lon, st_lat, marker='*', color='black', markersize=10, transform=ccrs.PlateCarree(), zorder=6)

        # Annotations
        name_display = station.capitalize()
        ax.set_title(rf"$\mathbf{{{PANEL_LABELS[i]}}}$ {name_display}", loc="left", fontsize=11)
        
        textstr = (f"Station: ({st_lat:.4f}, {st_lon:.4f})\n"
                   f"Selected pixel: ({px_lat:.4f}, {px_lon:.4f})\n"
                   f"Distance: {row['distance_km']:.2f} km\n"
                   f"Grid index: ({row['grid_row']}, {row['grid_column']})")
        
        props = dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='#cccccc')
        ax.text(0.03, 0.03, textstr, transform=ax.transAxes, fontsize=8, verticalalignment='bottom', bbox=props, zorder=7)

    # ── Legend and Summary Table ──────────────────────────────────────────────
    ax_legend = fig.add_subplot(gs[1, 3])
    ax_legend.axis("off")
    
    # Legend
    handles = [
        mlines.Line2D([], [], color='black', marker='*', linestyle='None', markersize=10, label='Station location'),
        mlines.Line2D([], [], color='orangered', marker='o', linestyle='None', markersize=6, label='Selected pixel centre'),
        mpatches.Patch(facecolor='orange', edgecolor='orange', alpha=0.2, label='Selected grid cell'),
        mlines.Line2D([], [], color='red', linestyle='--', linewidth=1.2, label='Station–pixel connection'),
        mlines.Line2D([], [], color='#999999', marker='+', linestyle='None', markersize=4, label='Other grid-cell centres')
    ]
    ax_legend.legend(handles=handles, loc='upper center', title="Legend", fontsize=9, title_fontsize=10, framealpha=1.0)
    
    # Table below legend
    col_labels = ["Station", "Dist (km)", "Flag"]
    table_data = [[row["station_name"].capitalize(), f"{row['distance_km']:.2f}", row["distance_flag"]] for _, row in df_out.iterrows()]
    table = ax_legend.table(cellText=table_data, colLabels=col_labels, loc='lower center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    
    # Export
    png_path = os.path.join(out_dir, "nearestPointStaion.png")
    
    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Figure saved to: {png_path}")
    
    # Final terminal summary
    print("\n==================================================")
    print("FINAL STATION-TO-PIXEL DISTANCE SUMMARY")
    print("==================================================")
    max_dist = 0
    max_station = ""
    for _, row in df_out.iterrows():
        print(f"{row['station_name']:12s} : {row['distance_km']:>6.2f} km [{row['distance_flag']}]")
        if row['distance_km'] > max_dist:
            max_dist = row['distance_km']
            max_station = row['station_name']
    
    print("\nRESULT: ", end="")
    if max_station == "TANGER":
        print(f"Tanger has the largest station-to-pixel distance ({max_dist:.2f} km).")
    else:
        print(f"{max_station} has the largest station-to-pixel distance ({max_dist:.2f} km), not Tanger.")
    print("==================================================\n")

if __name__ == "__main__":
    main()
