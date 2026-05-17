"""
==========================================================
 Script: glue_regional_predictions.py
 Author: Antigravity AI
 Description:
     Spatially concatenates (glues) North and South regional NetCDF
     predictions along the latitude dimension for CNN and ViT models.
     This compiles the divided sub-regions back into a unified Morocco
     grid (lat: 21.0 to 37.0, lon: -18.0 to 0.0) for testing (2006-2020),
     allowing direct metric comparisons against the unified Moroccan baselines.
==========================================================
"""

import os
import xarray as xr
import numpy as np

def glue_predictions():
    base_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main"
    results_dir = os.path.join(base_dir, "results")
    
    # Models to process (excluding GLM since it fits pixel-wise and yields identical parameters)
    models = [
        {
            "name": "cnn_exp3",
            "north_dir": "cnn_exp3_north/region_lat_28.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "south_dir": "cnn_exp3_south/region_lat_21.0_28.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "target_dir": "cnn_exp3_regional_glued/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "file_name": "cnn_predictions_era5_to_mswep.nc",
            "variable": "precipitation"
        },
        {
            "name": "cnn_exp5",
            "north_dir": "cnn_exp5_north/region_lat_28.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "south_dir": "cnn_exp5_south/region_lat_21.0_28.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "target_dir": "cnn_exp5_regional_glued/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "file_name": "cnn_predictions_era5_to_mswep.nc",
            "variable": "precipitation"
        },
        {
            "name": "vit_best",
            "north_dir": "vit_precip_exp21_best_hybrid_north/region_lat_28.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn",
            "south_dir": "vit_precip_exp21_best_hybrid_south/region_lat_21.0_28.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn",
            "target_dir": "vit_regional_glued/region_lat_21.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn",
            "file_name": "vit_predictions_era5_to_mswep.nc",
            "variable": "precipitation"
        },
        {
            "name": "glm_precip_l2",
            "north_dir": "glm_precip_l2_north/region_lat_28.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_10ep_wd",
            "south_dir": "glm_precip_l2_south/region_lat_21.0_28.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_10ep_wd",
            "target_dir": "glm_precip_l2_regional_glued/region_lat_21.0_36.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_10ep_wd",
            "file_name": "glm_predictions_era5_to_mswep.nc",
            "variable": "precipitation"
        }
    ]
    
    print("==================================================================")
    print("      Spatially Concatenating Regional downscaling predictions")
    print("==================================================================")
    
    for model in models:
        north_path = os.path.join(results_dir, model["north_dir"], "output_data", model["file_name"])
        south_path = os.path.join(results_dir, model["south_dir"], "output_data", model["file_name"])
        target_path_dir = os.path.join(results_dir, model["target_dir"], "output_data")
        target_file_path = os.path.join(target_path_dir, model["file_name"])
        
        print(f"\nProcessing model: {model['name']}")
        
        if not os.path.exists(north_path):
            print(f"  [PENDING] North predictions not found yet: {north_path}")
            continue
        if not os.path.exists(south_path):
            print(f"  [PENDING] South predictions not found yet: {south_path}")
            continue
            
        print("  Loading sub-regional NetCDF files...")
        ds_n = xr.open_dataset(north_path)
        ds_s = xr.open_dataset(south_path)
        
        print(f"  - North Shape: {ds_n[model['variable']].shape} (lat range: {float(ds_n.lat.min()):.2f} to {float(ds_n.lat.max()):.2f})")
        print(f"  - South Shape: {ds_s[model['variable']].shape} (lat range: {float(ds_s.lat.min()):.2f} to {float(ds_s.lat.max()):.2f})")
        
        # Spatial concatenation along 'lat' dimension
        print("  Concatenating along 'lat' dimension...")
        ds_glued = xr.concat([ds_s, ds_n], dim="lat")
        
        # Sort along lat to be absolutely clean
        ds_glued = ds_glued.sortby("lat")
        
        # Create output directories
        os.makedirs(target_path_dir, exist_ok=True)
        
        # Save output NetCDF
        print(f"  Saving glued NetCDF: {target_file_path}")
        ds_glued.to_netcdf(target_file_path)
        
        print(f"  - Glued Shape: {ds_glued[model['variable']].shape} (lat range: {float(ds_glued.lat.min()):.2f} to {float(ds_glued.lat.max()):.2f})")
        print(f"  [SUCCESS] Spliced regional models for {model['name']} compiled successfully!")
        
    print("\n==================================================================")
    print("      Regional split assembly pipeline complete.")
    print("==================================================================")

if __name__ == "__main__":
    glue_predictions()
