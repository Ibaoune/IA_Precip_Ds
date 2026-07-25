"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""

import os
import xarray as xr
import numpy as np

# File paths
lmdz35_raw_path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/data/lmdz/r35/hist/present/all_Mor/precip-hist.nc"
lmdz250_raw_path = "/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/shared/TEAM/data/lmdz/r250/amip/present/all_Mor/precip-amip.nc"

# Predictions paths (1979-2014)
pred_root = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/inference/results/1979_2014"
unet_l250_path = os.path.join(pred_root, "cnn_exp3/cnn_lmdz_250_present_true.nc")
cnn_l250_path = os.path.join(pred_root, "cnn_exp5/cnn_lmdz_250_present_true.nc")
glm_l250_path = os.path.join(pred_root, "glm_precip_l2/glm_lmdz_250_present_true.nc")
vit_l250_path = os.path.join(pred_root, "vit_precip_exp21_best_hybrid/vit_lmdz_250_present_true.nc")

unet_l35_path = os.path.join(pred_root, "cnn_exp3/cnn_lmdz_35_present_true.nc")
cnn_l35_path = os.path.join(pred_root, "cnn_exp5/cnn_lmdz_35_present_true.nc")
glm_l35_path = os.path.join(pred_root, "glm_precip_l2/glm_lmdz_35_present_true.nc")

# Output directory
os.makedirs("datasets/coarse_lmdz250", exist_ok=True)
os.makedirs("datasets/coarse_lmdz35", exist_ok=True)

# Load a prediction to get the target 10km grid
print("Loading 10km target grid...")
pred_10km = xr.open_dataset(unet_l250_path)
target_lat = pred_10km["lat"]
target_lon = pred_10km["lon"]
target_time = pred_10km["time"]

# Function to process raw LMDZ
def process_raw_lmdz(raw_path, label):
 print(f"Processing raw {label}...")
 ds = xr.open_dataset(raw_path)
 
 # 1. Normalize time counter to daily dates (midnight)
 time_normalized = ds["time_counter"].dt.floor("D")
 ds = ds.assign_coords(time_counter=time_normalized)
 ds = ds.rename({"time_counter": "time"})
 
 # 2. Rename variable precip to precipitation
 if "precip" in ds:
 ds = ds.rename({"precip": "precipitation"})
 
 # 3. Unit conversion: kg/(s*m2) -> mm/day (multiply by 86400)
 ds["precipitation"] = ds["precipitation"] * 86400.0
 ds["precipitation"].attrs["units"] = "mm/day"
 ds["precipitation"].attrs["long_name"] = "Precipitation"
 
 # Keep only the precipitation variable
 ds = ds[["precipitation"]]
 return ds

# Prepare raw 10km interpolated datasets
ds_l250_raw = process_raw_lmdz(lmdz250_raw_path, "LMDZ250")
ds_l35_raw = process_raw_lmdz(lmdz35_raw_path, "LMDZ35")

# Interpolate raw datasets to 10km grid
print("Interpolating LMDZ250 raw to 10km...")
ds_l250_10km = ds_l250_raw.interp(lat=target_lat, lon=target_lon, method="linear")
ds_l250_10km.to_netcdf("datasets/raw_lmdz250_10km.nc")
print("Saved: datasets/raw_lmdz250_10km.nc")

print("Interpolating LMDZ35 raw to 10km...")
ds_l35_10km = ds_l35_raw.interp(lat=target_lat, lon=target_lon, method="linear")
ds_l35_10km.to_netcdf("datasets/raw_lmdz35_10km.nc")
print("Saved: datasets/raw_lmdz35_10km.nc")

# Prepare coarse prediction datasets (interpolated from 10km down to coarse grids)
def coarsen_and_save(src_path, dest_path, target_grid_ds, label):
 print(f"Coarsening {label} to original coarse grid...")
 pred = xr.open_dataset(src_path)
 
 # Interpolate from 10km down to coarse lat/lon
 pred_coarse = pred.interp(lat=target_grid_ds["lat"], lon=target_grid_ds["lon"], method="linear")
 
 # Align time dimension exactly with target grid (normalize time if needed)
 pred_coarse = pred_coarse.sel(time=slice("1979-01-01", "2014-12-31"))
 pred_coarse.to_netcdf(dest_path)
 print(f"Saved: {dest_path}")

# LMDZ 250 predictions coarse
coarsen_and_save(unet_l250_path, "datasets/coarse_lmdz250/unet_lmdz250_coarse.nc", ds_l250_raw, "UNET LMDZ250")
coarsen_and_save(cnn_l250_path, "datasets/coarse_lmdz250/cnn_lmdz250_coarse.nc", ds_l250_raw, "CNN LMDZ250")
coarsen_and_save(glm_l250_path, "datasets/coarse_lmdz250/glm_lmdz250_coarse.nc", ds_l250_raw, "GLM LMDZ250")
coarsen_and_save(vit_l250_path, "datasets/coarse_lmdz250/vit_lmdz250_coarse.nc", ds_l250_raw, "ViT LMDZ250")

# LMDZ 35 predictions coarse
coarsen_and_save(unet_l35_path, "datasets/coarse_lmdz35/unet_lmdz35_coarse.nc", ds_l35_raw, "UNET LMDZ35")
coarsen_and_save(cnn_l35_path, "datasets/coarse_lmdz35/cnn_lmdz35_coarse.nc", ds_l35_raw, "CNN LMDZ35")
coarsen_and_save(glm_l35_path, "datasets/coarse_lmdz35/glm_lmdz35_coarse.nc", ds_l35_raw, "GLM LMDZ35")

print("All comparison datasets prepared successfully!")
