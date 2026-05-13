# Documentation for `src/data/interpolation.py`

## Overview
==========================================================
 Script: interpolation.py
 Description:
     Contains functions to interpolate predictor datasets
     (e.g., ERA5 or LMDZ) to a 2-degree grid.
==========================================================

## Functions
### `def interpolate_to_target_resolution(...)`
Interpolates an xarray Dataset or DataArray to a target degree resolution.

Args:
    ds: xarray.Dataset or xarray.DataArray to be interpolated.
    resolution: Target grid resolution in degrees.
    lon_name: Name of the longitude coordinate.
    lat_name: Name of the latitude coordinate.
    method: Interpolation method ('linear', 'nearest').
    bounds: Optional tuple (min_lon, max_lon, min_lat, max_lat). If provided,
            the target grid is anchored to the floor/ceil of these bounds.

## How to Modify
If you need to make changes to `interpolation.py`:
- **Logic changes**: Locate the corresponding function. Update its docstring if you change its signature or behavior.
- Ensure any related imports in other files are updated if you change function/class names.