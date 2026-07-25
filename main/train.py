"""
==========================================================
 Script: train.py
 Author: M. El Aabaribaoune (@um6p)
 Description:
     Main entry point for training the downscaling model.
     - Loads configuration
     - Loads and preprocesses data
     - Trains the model
     - Saves trained weights and losses

 Notes:
     - Compatible CPU / GPU
     - Model-agnostic (ViT by default)
==========================================================
"""

import torch
import sys
import os

from  src.core.config import load_config 
from src.data.data_loading import load_datasets
from src.data.preprocessing import preprocess_data
from src.core.training import train_model
from src.core.utils import vprint, save_model, set_verbose

def main():
    # ----------------
    # Load configuration
    # ----------------
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    cfg = load_config(train_mode=True, path=cfg_path)
    set_verbose(cfg.verbose)
    
    vprint(f"Using device: {cfg.device}")
    vprint(f"=== Starting training process for {cfg.variable} ===")

    # ----------------
    # Load and preprocess data
    # ----------------
    # Load datasets returns: X, y_train, y_test, lon_in, lat_in, lon_out, lat_out, time_train, time_test
    datasets = load_datasets(cfg)
    X, y_train, y_test = datasets[0], datasets[1], datasets[2]
    lon_out, lat_out = datasets[5], datasets[6]
    time_train, time_test = datasets[7], datasets[8]

    # Preprocess data returns: x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor
    x_train_tensor, _, y_train_tensor, _ = preprocess_data(
        cfg, X, y_train, y_test, time_train=time_train, time_test=time_test
    )

    # GPU ADAPTATION: Ensure tensors are on the correct device.
    # Moving the full training dataset to the target device (e.g., GPU) here 
    # speeds up training if the dataset fits entirely in VRAM.
    x_train_tensor = x_train_tensor.to(cfg.device)
    y_train_tensor = y_train_tensor.to(cfg.device)

    # ----------------
    # Compute Land/Sea Mask
    # ----------------
    import regionmask
    import xarray as xr
    import numpy as np

    da = xr.DataArray(
        np.zeros((len(lat_out), len(lon_out))),
        coords=[("lat", lat_out), ("lon", lon_out)]
    )
    region = regionmask.defined_regions.natural_earth_v5_0_0.land_110
    mask = region.mask(da)
    land_mask_np = mask.notnull().values.astype("float32") # 1.0 for land, 0.0 for ocean

    if getattr(cfg, "loss_mask_enable", False) and getattr(cfg, "loss_mask_shapefile", None):
        vprint(f"[INFO] Applying sub-domain loss mask from: {cfg.loss_mask_shapefile}")
        import geopandas as gpd
        import pandas as pd
        if isinstance(cfg.loss_mask_shapefile, list):
            gdfs = [gpd.read_file(shp).to_crs("EPSG:4326") for shp in cfg.loss_mask_shapefile]
            gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
        else:
            gdf = gpd.read_file(cfg.loss_mask_shapefile).to_crs("EPSG:4326")
        gdf_dissolved = gdf.dissolve()
        gdf_dissolved["name"] = ["mask_region"]
        gdf_dissolved = gdf_dissolved.reset_index(drop=True)
        rm_mask = regionmask.from_geopandas(gdf_dissolved, names="name", name="mask_region")
        sub_mask_np = rm_mask.mask(da)
        region_mask_np = (~sub_mask_np.isnull()).values.astype("float32")
        final_mask_np = land_mask_np * region_mask_np
    else:
        final_mask_np = land_mask_np

    land_mask_tensor = torch.tensor(final_mask_np, dtype=torch.float32).to(cfg.device)

    # ----------------
    # Train model
    # ----------------
    model, train_losses, val_losses = train_model(
        cfg, x_train_tensor, y_train_tensor, land_mask=land_mask_tensor
    )

    # ----------------
    # Save model
    # ----------------
    # save_model handled by utils.py, saves to results/<exp>/models/ and saves config as well
    model_path = save_model(cfg, model, train_losses, val_losses)
    vprint(f"Train execution over. Results saved in {cfg.exp_dir}")

    vprint("=== Training completed ===")


if __name__ == "__main__":
    main()
