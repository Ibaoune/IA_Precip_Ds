"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""


import os
import sys
import xarray as xr
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import src.utils as utils

RESULTS_DIR = "results/eval_unet_experiments/2006-01-01_2020-12-31/allmorr/test/bias/results"
FIGURES_DIR = "results/eval_unet_experiments/2006-01-01_2020-12-31/allmorr/test/bias/figures/all"
os.makedirs(FIGURES_DIR, exist_ok=True)

models = ["CNN_Base"] + [f"UNet_Exp{i}" for i in range(1, 11)]

# Load the data
error_dict = {}
for m in models:
    file_path = os.path.join(RESULTS_DIR, f"{m}_pr_allmorr_calcul_land_strategy_mean_first_corr_per_year_Annual.nc")
    if os.path.exists(file_path):
        ds = xr.open_dataset(file_path)
        # some dimensions might have 'year', take the mean
        if "year" in ds.dims:
            ds = ds.mean("year")
        # Extract the variable (it's either named 'pr' or 'precipitation' or the metric itself)
        # Let's just grab the first data variable
        var_name = list(ds.data_vars)[0]
        error_dict[m] = ds[var_name]
    else:
        print(f"Missing {file_path}")

# Split into two groups
group1_names = ["CNN_Base"] + [f"UNet_Exp{i}" for i in range(1, 6)]
group2_names = ["CNN_Base"] + [f"UNet_Exp{i}" for i in range(6, 11)]

dict1 = {k: error_dict[k] for k in group1_names if k in error_dict}
dict2 = {k: error_dict[k] for k in group2_names if k in error_dict}

# Plot Group 1
utils.plot_spatial_maps(
    dict1, "bias", period="Annual",
    save_path=os.path.join(FIGURES_DIR, "spatial_bias_error_Annual_Exp1_5.png"),
    title="Annual Spatial Bias Error (CNN vs UNet Exp1-5)",
    unit="mm/day", nrows=2
)

# Plot Group 2
utils.plot_spatial_maps(
    dict2, "bias", period="Annual",
    save_path=os.path.join(FIGURES_DIR, "spatial_bias_error_Annual_Exp6_10.png"),
    title="Annual Spatial Bias Error (CNN vs UNet Exp6-10)",
    unit="mm/day", nrows=2
)

print("Custom bias figures generated successfully!")
