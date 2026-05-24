# Documentation for `src/core/utils.py`

## Overview
==========================================================
 Script: utils.py
 Author: M. El Aabaribaoune (@um6p)
 Description:
     Collection of utility and helper functions used across
     the project, including:
     - Verbose printing utilities
     - Dataset spatial masking
     - Visualization and comparison plots
     - Loss curve plotting
     - Experiment metadata formatting
     - Intelligent experiment path construction
     - Model saving and loading utilities

==========================================================

## Functions
### `def set_verbose(...)`
No description.

### `def vprint(...)`
No description.

### `def mask_dataset(...)`
Apply spatial mask and print detailed debug information.

### `def spatial_comparaison_plot(...)`
Plot spatial comparison between reference dataset and model output.

### `def monthly_precip_comparaison_plot(...)`
Plot monthly averaged precipitation comparison and compute bias / RMSE.

### `def plot_losses(...)`
Plot training and validation loss curves.

### `def format_components_for_title(...)`
Compact plot title formatter (max 4 lines)
Designed for evaluation plots.

### `def build_experiment_path(...)`
Build a structured experiment path. 
Uses the hierarchical path defined in cfg.exp_dir.

### `def save_model(...)`
No description.

### `def load_model(...)`
Load trained model checkpoint.
Supports both torch (.pth) and GLM (.pkl) models.

### `def calculate_rmse(...)`
Compute RMSE along the time axis (axis 0).
Accepts 1-D (single grid point) or N-D arrays.

### `def calculate_r2(...)`
Compute R² (coefficient of determination) along the time axis.

### `def calculate_pearson(...)`
Compute Pearson correlation coefficient along the time axis
for each spatial grid point.

### `def compute_spatial_metric(...)`
Apply a metric function to compute a 2-D spatial map.
metric_fn should accept (y_true, y_pred) with shape (time, lat, lon)
and return a (lat, lon) map.

### `def calculate_monthly_metrics(...)`
Compute monthly Pearson correlation, RMSE, and R² between
two xarray DataArrays (model predictions and observations).

Returns:
    months       : array of month indices (1..12)
    monthly_corr : Pearson correlation per month
    monthly_rmse : RMSE per month
    monthly_r2   : R² per month

### `def get_machine_features(...)`
Returns a dictionary of relevant machine features (CPU, RAM, GPU).

### `def estimate_total_time(...)`
Given the duration of one epoch, estimate the total time for all epochs.
Returns a formatted string (HH:MM:SS or MM:SS).

## How to Modify
If you need to make changes to `utils.py`:
- **Logic changes**: Locate the corresponding function. Update its docstring if you change its signature or behavior.
- Ensure any related imports in other files are updated if you change function/class names.