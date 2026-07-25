"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""


import os
import xarray as xr
import numpy as np
import geopandas as gpd
import regionmask

def main():
    base_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling"
    main_dir = os.path.join(base_dir, "main")
    postproc_dir = os.path.join(base_dir, "postproc")
    results_dir = os.path.join(main_dir, "results")
    
    # Shapefile paths
    shape_files = {
        "north": os.path.join(postproc_dir, "shape_files", "north.shp"),
        "northeast": os.path.join(postproc_dir, "shape_files", "north_east.shp"),
        "east": os.path.join(postproc_dir, "shape_files", "east.shp"),
        "south": os.path.join(postproc_dir, "shape_files", "south.shp"),
    }
    
    # We will glue:
    # 1. cnn_exp3_lossmask (Scenario 2)
    # 2. cnn_exp5_lossmask (Scenario 2)
    # 3. vit_lossmask (Scenario 2)
    # 4. cnn_exp3_scenario3 (Scenario 3)
    # 5. cnn_exp5_scenario3 (Scenario 3)
    # 6. vit_scenario3 (Scenario 3)

    jobs = [
        # --- Scenario 2 (Loss Masking) ---
        {
            "name": "cnn_exp3_lossmask",
            "type": "scenario2",
            "file_name": "cnn_predictions_era5_to_mswep.nc",
            "lat_range": "region_lat_21.0_37.0_lon_-18.0_0.0",
            "params": "gridbox_0.001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "regions": {
                "north": "cnn_exp3_lossmask_north",
                "northeast": "cnn_exp3_lossmask_north_east",
                "east": "cnn_exp3_lossmask_east",
                "south": "cnn_exp3_lossmask_south",
            },
            "target": "cnn_exp3_lossmask_glued"
        },
        {
            "name": "cnn_exp5_lossmask",
            "type": "scenario2",
            "file_name": "cnn_predictions_era5_to_mswep.nc",
            "lat_range": "region_lat_21.0_37.0_lon_-18.0_0.0",
            "params": "gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "regions": {
                "north": "cnn_exp5_lossmask_north",
                "northeast": "cnn_exp5_lossmask_north_east",
                "east": "cnn_exp5_lossmask_east",
                "south": "cnn_exp5_lossmask_south",
            },
            "target": "cnn_exp5_lossmask_glued"
        },
        {
            "name": "vit_lossmask",
            "type": "scenario2",
            "file_name": "vit_predictions_era5_to_mswep.nc",
            "lat_range": "region_lat_21.0_36.0_lon_-18.0_0.0",
            "params": "global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn",
            "regions": {
                "north": "vit_precip_exp21_best_hybrid_lossmask_north",
                "northeast": "vit_precip_exp21_best_hybrid_lossmask_north_east",
                "east": "vit_precip_exp21_best_hybrid_lossmask_east",
                "south": "vit_precip_exp21_best_hybrid_lossmask_south",
            },
            "target": "vit_precip_exp21_best_hybrid_lossmask_glued"
        },
        # --- Scenario 3 (Merged Loss Masking) ---
        {
            "name": "cnn_exp3_scenario3",
            "type": "scenario3",
            "file_name": "cnn_predictions_era5_to_mswep.nc",
            "lat_range": "region_lat_21.0_37.0_lon_-18.0_0.0",
            "params": "gridbox_0.001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "regions": {
                "north": "cnnexp3_scenario3_north_northeast",
                "northeast": "cnnexp3_scenario3_north_northeast",
                "east": "cnnexp3_scenario3_east_south",
                "south": "cnnexp3_scenario3_east_south",
            },
            "target": "cnnexp3_scenario3_glued"
        },
        {
            "name": "cnn_exp5_scenario3",
            "type": "scenario3",
            "file_name": "cnn_predictions_era5_to_mswep.nc",
            "lat_range": "region_lat_21.0_37.0_lon_-18.0_0.0",
            "params": "gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn",
            "regions": {
                "north": "cnnexp5_scenario3_north_northeast",
                "northeast": "cnnexp5_scenario3_north_northeast",
                "east": "cnnexp5_scenario3_east_south",
                "south": "cnnexp5_scenario3_east_south",
            },
            "target": "cnnexp5_scenario3_glued"
        },
        {
            "name": "vit_scenario3",
            "type": "scenario3",
            "file_name": "vit_predictions_era5_to_mswep.nc",
            "lat_range": "region_lat_21.0_36.0_lon_-18.0_0.0",
            "params": "global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn",
            "regions": {
                "north": "vit_scenario3_north_northeast",
                "northeast": "vit_scenario3_north_northeast",
                "east": "vit_scenario3_east_south",
                "south": "vit_scenario3_east_south",
            },
            "target": "vit_scenario3_glued"
        }
    ]

    for job in jobs:
        print(f"\n==========================================")
        print(f"Gluing: {job['name']}")
        print(f"==========================================")
        
        # Load datasets
        datasets = {}
        for r_name, dir_name in job["regions"].items():
            path = os.path.join(
                results_dir, 
                dir_name, 
                job["lat_range"], 
                "train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31", 
                job["params"], 
                "output_data", 
                job["file_name"]
            )
            if not os.path.exists(path):
                print(f"  [ERROR] File not found: {path}")
                continue
            datasets[r_name] = xr.open_dataset(path)
            
        if len(datasets) < 4:
            print(f"  [SKIPPING] Not all regional datasets were successfully loaded for {job['name']}")
            continue
            
        # We start with the 'north' dataset as baseline to keep coordinates/variables intact
        glued_ds = datasets["north"].copy(deep=True)
        var_name = "precipitation" if "precipitation" in glued_ds else "precip"
        
        # Create masks for each region
        for r_name, ds in datasets.items():
            shp_path = shape_files[r_name]
            gdf = gpd.read_file(shp_path).to_crs("EPSG:4326").dissolve()
            mask = regionmask.from_geopandas(gdf).mask(glued_ds.lon, glued_ds.lat)
            is_in_region = ~mask.isnull()
            
            # Update values inside this region's mask
            glued_ds[var_name] = xr.where(
                is_in_region, 
                ds[var_name], 
                glued_ds[var_name]
            ).transpose(*glued_ds[var_name].dims)
            print(f"  Applied spatial mask for region: {r_name}")
            
        # Target path
        target_path_dir = os.path.join(
            results_dir, 
            job["target"], 
            job["lat_range"], 
            "train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31", 
            job["params"], 
            "output_data"
        )
        os.makedirs(target_path_dir, exist_ok=True)
        target_path = os.path.join(target_path_dir, job["file_name"])
        
        glued_ds.to_netcdf(target_path)
        print(f"  [SUCCESS] Glued dataset saved to: {target_path}")

if __name__ == "__main__":
    main()
