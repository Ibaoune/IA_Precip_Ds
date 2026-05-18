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
        print(f"[INFO] Loading observations from Excel: {excel_path}...")
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

def extract_nearest_gridpoint(ds, lat, lon, variable_name=None):
    """
    Queries a NetCDF xarray dataset at the nearest gridpoint matching lat/lon coordinates.
    Correctly handles different latitude/longitude coordinate naming conventions (e.g. lat/lon, latitude/longitude).
    """
    lat_key = "lat" if "lat" in ds.coords else "latitude"
    lon_key = "lon" if "lon" in ds.coords else "longitude"
    
    # Query nearest grid cell
    point = ds.sel({lat_key: lat, lon_key: lon}, method="nearest")
    
    # Retrieve correct variable
    if variable_name is None:
        possible_vars = ["precipitation", "precip", "pr", "precipitation_flux"]
        for var in possible_vars:
            if var in point.data_vars:
                variable_name = var
                break
        if variable_name is None:
            raise KeyError(f"No precipitation variable found in dataset. Data vars: {list(point.data_vars)}")
            
    return point[variable_name]

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
