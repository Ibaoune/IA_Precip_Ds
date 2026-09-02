"""
Author: M. El Aabaribaoune (@um6p)
Description: Core utility module providing shared functions for data loading, spatial interpolation, and metric calculations.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np

DEFAULT_EXCEL_PATH = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/insitu/ABH_stations_plusziz_and_abhbc1.xlsx"
DEFAULT_CACHE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/stations_cache.csv"))

def load_insitu_observations(excel_path=DEFAULT_EXCEL_PATH, cache_path=DEFAULT_CACHE_PATH, start_date="2006-01-01", end_date="2020-12-31"):
    """
    Loads weather station daily observations from a pre-compiled CSV cache if available,
    otherwise loads from the Excel sheet, filters by date range, and extracts all station metadata (lat/lon).
    """
    # Resolve paths relative to the subproject root if they are relative
    if not os.path.isabs(cache_path):
        cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", cache_path))
    if not os.path.isabs(excel_path):
        excel_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", excel_path))

    if os.path.exists(cache_path):
        print(f"[INFO] Loading observations from fast CSV cache: {cache_path}...")
        df = pd.read_csv(cache_path)
    else:
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Observations file not found at: {excel_path}")
        print(f"[INFO] Loading observations from raw file: {excel_path}...")
        if excel_path.endswith('.csv'):
            df = pd.read_csv(excel_path)
        else:
            df = pd.read_excel(excel_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Filter by date range
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    df = df.sort_values("Date")
    print(f"[INFO] Filtered {len(df)} daily observation records between {start_date} and {end_date}.")
    return df

def get_station_metadata(df):
    """
    Extracts a dictionary of coordinates for each unique station in the dataframe.
    """
    stations_meta = {}
    for station in df["Station"].unique():
        st_data = df[df["Station"] == station]
        lat = st_data["Latitude"].iloc[0]
        lon = st_data["Longitude"].iloc[0]
        stations_meta[station] = {"lat": lat, "lon": lon}
    return stations_meta

def extract_nearest_gridpoint(ds, lat, lon, variable_name=None, search_radius=2):
    """
    Queries a NetCDF xarray dataset at the nearest gridpoint matching lat/lon coordinates.
    Correctly handles different latitude/longitude coordinate naming conventions.
    If search_radius > 0 and the nearest point is all NaNs, it will search the 
    surrounding neighborhood to find the closest valid (non-NaN) grid cell.
    """
    lat_key = "lat" if "lat" in ds.coords else "latitude"
    lon_key = "lon" if "lon" in ds.coords else "longitude"
    
    # Identify the variable first
    if variable_name is None:
        possible_vars = ["precipitation", "precip", "pr", "precipitation_flux"]
        for var in possible_vars:
            if var in ds.data_vars:
                variable_name = var
                break
        if variable_name is None:
            raise KeyError(f"No precipitation variable found in dataset. Data vars: {list(ds.data_vars)}")
            
    da = ds[variable_name]
    
    # --- TANGER OVERRIDE ---
    # The default nearest pixel for Tanger (35.72, -5.90) is (35.75, -5.95) which is mostly sea (34% land).
    # We force selection of the pixel to the East (35.75, -5.85) which is > 90% on land (94%).
    if abs(lat - 35.72) < 0.01 and abs(lon - -5.90) < 0.01:
        lat = 35.75
        lon = -5.85
        
    # --- DAKHLA OVERRIDE ---
    # Dakhla (23.72, -15.93) is on a narrow peninsula. The default nearest pixel is (23.75, -15.95) 
    # which is 38% land. The nearest pixel >90% land is on the mainland at (23.75, -15.75).
    if abs(lat - 23.72) < 0.01 and abs(lon - -15.93) < 0.01:
        lat = 23.75
        lon = -15.75
    # -----------------------
    
    # 1. Query strictly nearest grid cell
    point = da.sel({lat_key: lat, lon_key: lon}, method="nearest")
    
    # 2. If it's fully masked (NaNs) and a radius is allowed, search neighbors
    if search_radius > 0 and point.isnull().all().values:
        idx_lat = np.abs(ds[lat_key].values - lat).argmin()
        idx_lon = np.abs(ds[lon_key].values - lon).argmin()
        
        min_dist = float('inf')
        best_point = point
        
        for i in range(max(0, idx_lat - search_radius), min(len(ds[lat_key]), idx_lat + search_radius + 1)):
            for j in range(max(0, idx_lon - search_radius), min(len(ds[lon_key]), idx_lon + search_radius + 1)):
                if i == idx_lat and j == idx_lon:
                    continue # already checked the center
                    
                candidate = da.isel({lat_key: i, lon_key: j})
                if not candidate.isnull().all().values:
                    # Calculate simple distance proxy
                    c_lat = ds[lat_key].values[i]
                    c_lon = ds[lon_key].values[j]
                    dist = (c_lat - lat)**2 + (c_lon - lon)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_point = candidate
                        
        point = best_point
            
    return point

def align_series(obs_series, model_da):
    """
    Aligns and reindexes model predictions (xarray DataArray) to the observation series index,
    returning a clean DataFrame containing overlapping values without NaNs.
    """
    # Create pandas series with datetime index for prediction
    sim_series = pd.Series(
        model_da.values,
        index=pd.to_datetime(model_da["time"].values)
    )
    # Reindex to match observation dates exactly
    sim_aligned = sim_series.reindex(obs_series.index)
    
    merged = pd.concat([obs_series, sim_aligned], axis=1)
    merged.columns = ["obs", "sim"]
    return merged.dropna()
