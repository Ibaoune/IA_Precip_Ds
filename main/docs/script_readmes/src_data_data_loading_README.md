# Documentation for `src/data/data_loading.py`

## Overview
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

## Functions
### `def _is_daily_time_index(...)`
Returns True if the time_index spacing is (approximately) 1 day.
Uses median timestep for robustness to missing values.

### `def _print_basic_stats(...)`
Print mean / min / max statistics for an xarray DataArray.

### `def _rh_to_specific_humidity(...)`
Convert relative humidity (fraction 0-1) to specific humidity (kg/kg).
Uses Magnus formula for saturation vapor pressure.
rh : relative humidity, dimensionless fraction [0, 1]
t_k : temperature in Kelvin
p_hpa: pressure level in hPa (scalar)

### `def _process_level_array(...)`
Common processing for a single level array (renaming, transposition, interpolation).

### `def load_datasets(...)`
Unified loader for ERA5 → MSWEP and LMDZ → LMDZ35.

## How to Modify
If you need to make changes to `data_loading.py`:
- **Logic changes**: Locate the corresponding function. Update its docstring if you change its signature or behavior.
- Ensure any related imports in other files are updated if you change function/class names.