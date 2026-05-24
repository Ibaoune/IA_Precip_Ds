"""
==========================================================
 Script: data_loading.py
 Author: M. El Aabaribaoune (@um6p)
 Description:
 Unified data loader for downscaling experiments.

 - Loads predictors (ERA5 or LMDZ)
 - Loads precipitation targets (MSWEP or LMDZ35)
 - Applies spatial masking
 - Handles daily / sub-daily temporal resolution
 - Returns tensors and coordinate metadata

 Notes:
 - Model-agnostic (ViT, CNN, UNet, RF...)
 - GPU-ready but does NOT force GPU allocation
 - Logging via vprint only
==========================================================
"""

import os
import numpy as np
import xarray as xr
import torch

import utils as use
from utils import vprint
from interpolation import interpolate_to_target_resolution
# -------------------------------------
# Helpers
# -------------------------------------

def _is_daily_time_index(time_index):
 """
 Returns True if the time_index spacing is (approximately) 1 day.
 Uses median timestep for robustness to missing values.
 """
 if len(time_index) < 2:
 return True

 deltas = np.diff(time_index.values.astype("datetime64[ns]"))
 median_days = np.median(deltas).astype("timedelta64[ns]") / np.timedelta64(1, "D")
 return np.isclose(median_days, 1.0, rtol=1e-3, atol=1e-6)


def _print_basic_stats(arr, name, vprint_fn):
 """
 Print mean / min / max statistics for an xarray DataArray.
 """
 try:
 # Load once — calling .mean()/.min()/.max() separately on a lazy array
 # triggers three independent disk reads, tripling peak RAM for large arrays.
 vals = arr.values
 if not np.issubdtype(vals.dtype, np.number):
 vals = vals.astype("float32")

 flat = vals.ravel()
 if flat.size == 0:
 vprint_fn(f" [STATS] {name}: EMPTY array")
 return

 mean = np.nanmean(flat)
 vmin = np.nanmin(flat)
 vmax = np.nanmax(flat)
 units = arr.attrs.get("units", "")
 unit_str = f" {units}" if units else ""

 vprint_fn(
 f" [STATS] {name}: mean={mean:.6g}{unit_str}, "
 f"min={vmin:.6g}, max={vmax:.6g}"
 )

 except Exception as e:
 vprint_fn(f" [STATS] {name}: unable to compute stats ({e})")

def _rh_to_specific_humidity(rh, t_k, p_hpa):
 """
 Convert relative humidity (fraction 0-1) to specific humidity (kg/kg).
 Uses Magnus formula for saturation vapor pressure.
 rh : relative humidity, dimensionless fraction [0, 1]
 t_k : temperature in Kelvin
 p_hpa: pressure level in hPa (scalar)
 """
 t_c = t_k - 273.15
 e_s = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5)) # saturation vapour pressure [hPa]
 e = rh * e_s # actual vapour pressure [hPa]
 q = 0.622 * e / (p_hpa - 0.378 * e) # specific humidity [kg/kg]
 return q


def _fix_swapped_lat_lon(ds, filename):
 """
 Detect and fix severely malformed lat/lon labels in LMDZ files.
 Morocco lons are negative (-18 to 0); if "lat" coord has negative values
 and "lon" coord has positive values, the labels are swapped.
 """
 if "lat" in ds.coords and "lon" in ds.coords:
 lat_min_val = float(ds["lat"].min())
 lon_min_val = float(ds["lon"].min())
 if lat_min_val < 10 and lon_min_val > 10:
 vprint(f" WARNING: severely malformed lat/lon in {filename} — re-mapping physically")
 true_lat_max, true_lat_min = float(ds.lon.max()), float(ds.lon.min())
 true_lon_max, true_lon_min = float(ds.lat.max()), float(ds.lat.min())
 len_lat = ds.dims["lat"]
 len_lon = ds.dims["lon"]
 new_lat = np.linspace(true_lat_max, true_lat_min, len_lat)
 new_lon = np.linspace(true_lon_max, true_lon_min, len_lon)
 ds = ds.rename({"lat": "Latitude", "lon": "Longitude"})
 ds = ds.assign_coords(Latitude=new_lat, Longitude=new_lon)
 return ds.rename({"Latitude": "lat", "Longitude": "lon"})
 return ds


def _process_level_array(cfg, arr, var, lev, curr_time_dim, curr_lev_dim):
 """
 Common processing for a single level array (renaming, transposition, interpolation).
 """
 # ERA5 cleanup
 if cfg.src == "era5":
 rename = {}
 if "latitude" in arr.dims:
 rename["latitude"] = "lat"
 if "longitude" in arr.dims:
 rename["longitude"] = "lon"
 if rename:
 arr = arr.rename(rename)

 if "level" in arr.dims:
 arr = arr.drop_vars("level", errors="ignore")
 if "level" in arr.coords:
 arr = arr.drop_vars("level", errors="ignore")

 for c in ["level", "number", "expver", "surface", "valid_time"]:
 if c in arr.coords:
 arr = arr.drop_vars(c, errors="ignore")

 # Coordinate names
 lat_name = 'lat' if 'lat' in arr.coords or 'lat' in arr.dims else 'latitude'
 lon_name = 'lon' if 'lon' in arr.coords or 'lon' in arr.dims else 'longitude'
 
 rename_dict = {curr_time_dim: "time"}
 if lat_name != "lat": rename_dict[lat_name] = "lat"
 if lon_name != "lon": rename_dict[lon_name] = "lon"
 arr = arr.rename(rename_dict)

 # Explicitly drop the level coordinate to avoid MergeError during concat
 if curr_lev_dim in arr.coords:
 arr = arr.drop_vars(curr_lev_dim)

 arr = arr.transpose("time", "lat", "lon")
 
 # Subset dates
 time_slice = slice(
 min(cfg.start_date_train, cfg.start_date_test),
 max(cfg.end_date_train, cfg.end_date_test)
 )
 arr_sub = arr.sel({"time": time_slice})
 
 # Interpolate
 
 arr_sub = interpolate_to_target_resolution(
 arr_sub, 
 resolution=cfg.resolution, 
 method=cfg.interpolation_type,
 bounds=(cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max)
 )
 
 _print_basic_stats(arr_sub, f"{var}_{lev}", vprint)
 return arr_sub.expand_dims({"level": [f"{var}_{lev}"]})

# -------------------------------------
# Main loader
# -------------------------------------

def load_datasets(cfg):
 """
 Unified loader for ERA5 → MSWEP and LMDZ → LMDZ35.
 """

 vprint(f"=== Loading datasets (src={cfg.src}, target={cfg.target}) ===")

 # 1) Load target variable
 if cfg.variable == "precip":
 pr_file = cfg.target_path
 if cfg.target == "lmdz35":
 precip_var = "precip"
 time_dim = "time_counter"
 elif cfg.target == "mswep":
 precip_var = "precipitation"
 time_dim = "time"
 else:
 raise ValueError(f"Unsupported precip target: {cfg.target}")
 elif cfg.variable == "temp":
 pr_file = cfg.target_path
 ds_temp_check = xr.open_dataset(pr_file)
 ds_temp_check = _fix_swapped_lat_lon(ds_temp_check, pr_file)
 if "air_temperature" in ds_temp_check.data_vars:
 precip_var = "air_temperature"
 elif "t2m" in ds_temp_check.data_vars:
 precip_var = "t2m"
 else:
 precip_var = "tas"
 
 time_dim = "time" if "time" in ds_temp_check.dims else "time_counter"
 ds_temp_check.close()
 else:
 raise ValueError(f"Unsupported variable: {cfg.variable}")

 vprint(f"Loading precipitation file: {pr_file}")
 if not os.path.exists(pr_file):
 if not cfg.train_mode:
 vprint(f" WARNING: Target file {pr_file} NOT FOUND. Creating dummy target for prediction.")
 ds_pr = None
 else:
 raise FileNotFoundError(f"Target file {pr_file} not found and train_mode is True.")
 else:
 ds_pr = xr.open_dataset(pr_file)
 ds_pr = _fix_swapped_lat_lon(ds_pr, pr_file)

 if ds_pr is not None:
 ds_pr = use.mask_dataset(
 ds_pr,
 slice(cfg.lon_min, cfg.lon_max),
 slice(cfg.lat_min, cfg.lat_max),
 )

 if precip_var not in ds_pr:
 raise KeyError(f"Precip variable '{precip_var}' not found")

 y_train_x = ds_pr[precip_var].sel(
 {time_dim: slice(cfg.start_date_train, cfg.end_date_train)}
 ).sortby("lat").sortby("lon")
 y_test_x = ds_pr[precip_var].sel(
 {time_dim: slice(cfg.start_date_test, cfg.end_date_test)}
 ).sortby("lat").sortby("lon")

 vprint(f"Precip shapes → train={y_train_x.shape}, test={y_test_x.shape}")

 if cfg.train_mode:
 # Resample to daily and load tensors only when training.
 # In inference mode, calling _is_daily_time_index(y_train_x) passes the full
 # 3D DataArray → .values loads 1+ GB of precipitation floats, interprets them
 # as datetime64 garbage, returns False, and triggers an unnecessary resample.
 if cfg.src == "era5":
 vprint("Checking target time resolution...")
 if not _is_daily_time_index(y_train_x.time):
 y_train_x = y_train_x.resample(time="1D").mean()
 if not _is_daily_time_index(y_test_x.time):
 y_test_x = y_test_x.resample(time="1D").mean()
 _print_basic_stats(y_train_x, "precip (train)", vprint)
 _print_basic_stats(y_test_x, "precip (test)", vprint)
 y_train = torch.tensor(y_train_x.values.astype("float32"))
 y_test = torch.tensor(y_test_x.values.astype("float32"))
 else:
 y_train = None
 y_test = None

 lon_out = ds_pr.lon.values
 lat_out = ds_pr.lat.values
 time_dim_found = next((d for d in [time_dim, "time", "time_counter"] if d in y_train_x.coords or d in y_train_x.dims), time_dim)
 time_train_out = y_train_x[time_dim_found].values # time coordinate only, tiny
 time_test_out = y_test_x[time_dim_found].values
 else:
 y_train, y_test, lon_out, lat_out, time_train_out, time_test_out = [None]*6

 # 2) Load predictor variables
 variables = cfg.variables
 levels = cfg.levels
 data_arrays = []
 lmdz_var_map = cfg.lmdz_var_map

 for var in variables:
 lmdz_var = lmdz_var_map.get(var.lower(), var.lower())
 
 if cfg.src == "lmdz":
 base_dir = getattr(cfg, "folder", cfg.bc_reference_folder).rstrip("/") + "/"
 suffix = "hist" 
 for s in ["ssp245", "ssp585"]:
 if s in base_dir.lower(): suffix = s; break
 file_pattern = cfg.lmdz_predictor_pattern
 else:
 base_dir = None
 suffix = None
 file_pattern = cfg.era5_predictor_pattern

 is_level_specific = "{level}" in file_pattern or "{lev}" in file_pattern

 if not is_level_specific:
 # Merged variable files
 if cfg.src == "lmdz":
 filename = file_pattern.format(folder=base_dir.rstrip("/"), lmdz_var=lmdz_var, suffix=suffix)
 else:
 filename = file_pattern.format(var=var.lower())

 vprint(f"Loading merged variable file: {filename}...")
 if not os.path.exists(filename):
 vprint(f" → FILE NOT FOUND: {filename}")
 continue

 try:
 ds_full = xr.open_dataset(filename)
 ds_full = _fix_swapped_lat_lon(ds_full, filename)
 curr_lev_dim = next((d for d in ["presnivs", "plev", "level"] if d in ds_full.dims), "level")
 curr_time_dim = next((d for d in ["time_counter", "time"] if d in ds_full.dims), "time")

 # For LMDZ 'q' stored as relative humidity, load temperature for RH→q conversion.
 # Skip if the humidity variable is already specific humidity (units "1" or "kg/kg").
 q_is_specific_humidity = False
 if cfg.src == "lmdz" and var.lower() == "q":
 _q_actual = next(
 (v for v in ds_full.data_vars if v.lower() == lmdz_var.lower()),
 next((v for v in ds_full.data_vars if curr_lev_dim in ds_full[v].dims), None),
 )
 if _q_actual:
 _units = ds_full[_q_actual].attrs.get("units", "").strip().lower()
 _lname = ds_full[_q_actual].attrs.get("long_name", "").lower()
 q_is_specific_humidity = _units in ("1", "kg/kg", "kg kg-1") or "specific" in _lname
 if q_is_specific_humidity:
 vprint(f" → '{_q_actual}' is already specific humidity (units='{_units}') — skipping RH→q conversion")

 temp_ds_masked = None
 temp_actual_var = None
 if cfg.src == "lmdz" and var.lower() == "q" and not q_is_specific_humidity:
 t_lmdz_var = lmdz_var_map.get("t", "temp")
 temp_filename = file_pattern.format(
 folder=base_dir.rstrip("/"), lmdz_var=t_lmdz_var, suffix=suffix
 )
 if os.path.exists(temp_filename):
 _temp_ds = xr.open_dataset(temp_filename)
 _temp_ds = _fix_swapped_lat_lon(_temp_ds, temp_filename)
 temp_ds_masked = use.mask_dataset(
 _temp_ds, slice(cfg.lon_min, cfg.lon_max), slice(cfg.lat_min, cfg.lat_max)
 )
 temp_actual_var = next(
 (v for v in temp_ds_masked.data_vars if v.lower() == t_lmdz_var.lower()),
 list(temp_ds_masked.data_vars)[0]
 )
 vprint(f" → Loaded temperature for RH→q conversion ({temp_filename})")
 else:
 vprint(f" → WARNING: RH→q conversion skipped — temp file missing: {temp_filename}")

 for lev in levels:
 vprint(f" → Extracting level {lev} from {var}")
 ds = use.mask_dataset(ds_full, slice(cfg.lon_min, cfg.lon_max), slice(cfg.lat_min, cfg.lat_max))
 actual_var = next(
 (v for v in ds.data_vars if v.lower() == lmdz_var.lower()),
 next((v for v in ds.data_vars if curr_lev_dim in ds[v].dims), list(ds.data_vars)[0])
 )
 target_lev = float(lev)
 if ds[curr_lev_dim].values.max() > 2000:
 target_lev = target_lev * 100.0

 arr = ds[actual_var].sel({curr_lev_dim: target_lev}, method="nearest").squeeze()

 # --- Geopotential / Geopotential Height (m -> m2/s2) ---
 if lmdz_var.lower() in ["geop", "zg"] or var.lower() == "z":
 units = arr.attrs.get("units", "").lower()
 if units == "m" or units == "meters":
 vprint(f" → Converting {actual_var} from m to m2/s2")
 arr = arr * 9.80665
 arr.attrs["units"] = "m2/s2"

 # Convert LMDZ relative humidity (fraction) → specific humidity (kg/kg)
 if cfg.src == "lmdz" and var.lower() == "q" and not q_is_specific_humidity and temp_ds_masked is not None:
 t_arr = temp_ds_masked[temp_actual_var].sel(
 {curr_lev_dim: target_lev}, method="nearest"
 ).squeeze()
 arr = _rh_to_specific_humidity(arr, t_arr, float(lev))
 vprint(f" → Converted rhum → specific humidity at {lev} hPa")

 p_arr = _process_level_array(cfg, arr, var, lev, curr_time_dim, curr_lev_dim)
 if p_arr is not None: data_arrays.append(p_arr)

 if temp_ds_masked is not None:
 _temp_ds.close()
 ds_full.close()
 except Exception as e:
 vprint(f" ERROR: {e}")

 else:
 # Level-specific files
 for lev in levels:
 if cfg.src == "lmdz":
 filename = file_pattern.format(folder=base_dir.rstrip("/"), lmdz_var=lmdz_var, suffix=suffix, level=lev, lev=lev)
 else:
 filename = file_pattern.format(var=var.lower(), level=lev, lev=lev)

 vprint(f"Loading level file: {filename}...")
 if not os.path.exists(filename):
 vprint(f" → FILE NOT FOUND: {filename}")
 continue

 try:
 ds = xr.open_dataset(filename).squeeze()
 ds = _fix_swapped_lat_lon(ds, filename)
 curr_time_dim = next((d for d in ["time_counter", "time"] if d in ds.dims), "time")
 curr_lev_dim = next((d for d in ["presnivs", "plev", "level"] if d in ds.dims), "level")
 
 ds = use.mask_dataset(ds, slice(cfg.lon_min, cfg.lon_max), slice(cfg.lat_min, cfg.lat_max))
 actual_var = next(
 (v for v in ds.data_vars if v.lower() == lmdz_var.lower()),
 next((v for v in ds.data_vars if curr_lev_dim in ds[v].dims), list(ds.data_vars)[0])
 )
 arr = ds[actual_var].squeeze()

 # --- Geopotential / Geopotential Height (m -> m2/s2) ---
 if lmdz_var.lower() in ["geop", "zg"] or var.lower() == "z":
 units = arr.attrs.get("units", "").lower()
 if units == "m" or units == "meters":
 vprint(f" → Converting {actual_var} from m to m2/s2")
 arr = arr * 9.80665
 arr.attrs["units"] = "m2/s2"

 p_arr = _process_level_array(cfg, arr, var, lev, curr_time_dim, curr_lev_dim)
 if p_arr is not None: data_arrays.append(p_arr)
 ds.close()
 except Exception as e:
 vprint(f" ERROR: {e}")



 if not data_arrays:
 raise ValueError("No predictor files loaded")

 X = xr.concat(data_arrays, dim="level", coords="minimal")
 X = X.transpose("time", "level", "lat", "lon")

 if cfg.src == "era5":
 vprint("Checking time resolution...")
 if not _is_daily_time_index(X.time):
 X = X.resample(time="1D").mean()

 x_train = X.sel(time=slice(cfg.start_date_train, cfg.end_date_train))
 x_test = X.sel(time=slice(cfg.start_date_test, cfg.end_date_test))

 # Fill metadata from X if it was missing from target
 if lon_out is None:
 lon_out = X.lon.values
 if lat_out is None:
 lat_out = X.lat.values
 if time_test_out is None:
 time_test_out = x_test.time.values
 if time_train_out is None:
 time_train_out = x_train.time.values
 
 # Fill dummy tensors only when training — predict.py never uses y_train/y_test
 # and each zeros((9497, 160, 180)) wastes ~1.1 GB per load_datasets call.
 if cfg.train_mode:
 if y_train is None:
 y_train = torch.zeros((len(time_train_out), len(lat_out), len(lon_out)))
 if y_test is None:
 y_test = torch.zeros((len(time_test_out), len(lat_out), len(lon_out)))

 vprint("=== Finished loading datasets ===")

 return (
 X,
 y_train,
 y_test,
 X.lon.values,
 X.lat.values,
 lon_out,
 lat_out,
 time_train_out,
 time_test_out,
 )

