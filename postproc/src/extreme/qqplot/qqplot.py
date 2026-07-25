"""
Author: M. El Aabaribaoune (@um6p)
Description: Computes and plots extreme precipitation indices.
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import warnings
from pathlib import Path

root_path = str(Path(__file__).resolve().parents[3])
if root_path not in sys.path:
    sys.path.append(root_path)

import utils

warnings.filterwarnings('ignore')

def compute_qq_quantiles(data, n_quantiles=10000, threshold=1):
    """
    Step 1 & 2 & 3: Data Flattening, Independent Sorting, and Quantile Sampling.
    """
    # 1. Data Flattening
    vals = data.values.flatten()
    
    # Filter valid values and wet days (Step 2 often focuses on rain distribution)
    vals = vals[~np.isnan(vals)]
    vals = vals[vals >= threshold]
    
    if len(vals) == 0:
        return np.full(n_quantiles, np.nan)
    
    # 2. Independent Sorting
    vals_sorted = np.sort(vals)
    
    # 3. Quantile Sampling
    # We want n_quantiles points from 0 to 1
    indices = np.linspace(0, len(vals_sorted) - 1, n_quantiles).astype(int)
    quantiles = vals_sorted[indices]
    
    return quantiles

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args, unknown = parser.parse_known_args()

    # === Load Configuration ===
    config_path = args.config if os.path.isabs(args.config) else os.path.join(os.path.dirname(__file__), args.config)
    config = utils.load_config(config_path)
    
    params = config['parameters']
    ref_cfg = config['reference']
    datasets_cfg = config['datasets']
    metric_cfg = config['metric']

    POST_PROCESS_ROOT = root_path
    
    # === Output Path Setup ===
    RESULTS_DIR = utils.get_results_dir(config, metric_cfg['name'], POST_PROCESS_ROOT)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    DATA_RESULTS_DIR = os.path.join(RESULTS_DIR, "results")
    os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
    
    region_name = params['region']
    threshold = metric_cfg.get('threshold', 1)
    n_quantiles = metric_cfg.get('n_quantiles', 10000)
    
    # === Load Reference ===
    print(f"[DEBUG] Loading reference: {ref_cfg['name']}...")
    ds_ref = xr.open_dataset(os.path.join(POST_PROCESS_ROOT, ref_cfg['file_path']))
    ds_ref = utils.mask_dataset(ds_ref, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
    ds_ref = ds_ref.sel(time=slice(params['start_date'], params['end_date']))
    var_ref = ds_ref[ref_cfg['variable_name']]
    
    # === Pre-load Datasets ===
    datasets_data = {}
    for dat in datasets_cfg:
        print(f"[DEBUG] Loading dataset: {dat['name']}...")
        ds_mod = xr.open_dataset(os.path.join(POST_PROCESS_ROOT, dat["file_path"]))
        ds_mod = utils.mask_dataset(ds_mod, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
        ds_mod = ds_mod.sel(time=slice(params['start_date'], params['end_date']))
        datasets_data[dat['name']] = ds_mod[dat["variable_name"]]
        
    mask_type = "land" if params['mask_land'] else "whole"
    _, months = utils.get_time_info(var_ref['time'].values)

    # === Process Each Season ===
    for season, season_months in utils.SEASONS.items():
        print(f"\nProcessing season: {season}")
        idx = np.isin(months, season_months)
        var_ref_s = var_ref.isel(time=idx)
        
        # Compute Reference Quantiles for this season
        ref_quantiles = compute_qq_quantiles(var_ref_s, n_quantiles=n_quantiles, threshold=threshold)
        
        results = {
            "quantile_rank": np.linspace(0, 1, n_quantiles),
            ref_cfg['name']: ref_quantiles
        }
        
        # Compute Model Quantiles for this season
        for name, var_mod in datasets_data.items():
            var_mod_s = var_mod.isel(time=idx)
            mod_quantiles = compute_qq_quantiles(var_mod_s, n_quantiles=n_quantiles, threshold=threshold)
            results[name] = mod_quantiles
            
        # Save Results
        df = pd.DataFrame(results)
        csv_filename = f"qqplot_data_{params['predictand']}_{region_name}_{mask_type}_{season}.csv"
        csv_path = os.path.join(DATA_RESULTS_DIR, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Saved QQ-plot data for {season} to: {csv_path}")

if __name__ == "__main__":
    main()
