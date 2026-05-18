import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# Add project root to sys.path
root_path = str(Path(__file__).resolve().parents[3])
if root_path not in sys.path:
    sys.path.append(root_path)

import utils

def compute_r01_frequency(dataset, ref=None, return_by_year=True, **kwargs):
    """
    Computes the frequency (percentage) of wet days where precipitation >= 1mm/day.
    
    Inputs:
        dataset (xr.DataArray): Daily precipitation.
        threshold (float): Threshold in mm/day (Default: 1.0).
        return_by_year (bool): If True, returns (year, lat, lon). If False, mean over all time.
    """
    threshold = kwargs.get('threshold', 1.0)
    # Create mask of wet days
    wet_mask = dataset >= threshold
    
    if return_by_year:
        # Group by year and calculate mean (fraction of wet days)
        # Multiply by 100 to get percentage
        r01 = wet_mask.groupby('time.year').mean(dim='time') * 100
        r01.name = "r01"
        return r01
    else:
        # Overall mean frequency
        return wet_mask.mean(dim='time') * 100

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args, unknown = parser.parse_known_args()

    # === Load Configuration ===
    config_path = args.config if os.path.isabs(args.config) else os.path.join(os.path.dirname(__file__), args.config)
    config = utils.load_config(config_path)
    
    params = config['parameters']
    metric_cfg = config['metric']
    ref_cfg = config['reference']
    datasets_cfg = config['datasets']

    METRIC = metric_cfg['name']
    STRATEGY = metric_cfg.get('strategy', 'mean_first')
    THRESHOLD = metric_cfg.get('threshold', 1.0)
    
    POST_PROCESS_ROOT = root_path
    
    # === Output Path Setup ===
    RESULTS_DIR = utils.get_results_dir(config, METRIC, POST_PROCESS_ROOT)
    DATA_RESULTS_DIR = os.path.join(RESULTS_DIR, "results")
    os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
    
    region_name = params['region']
    mask_type = "land" if params['mask_land'] else "whole"
    general_suffix = f"{params['predictand']}_{region_name}_calcul_{mask_type}"

    # === Load Reference ===
    print(f"Loading reference: {ref_cfg['name']}...")
    ds_ref = xr.open_dataset(os.path.join(POST_PROCESS_ROOT, ref_cfg['file_path']))
    ds_ref = utils.mask_dataset(ds_ref, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
    ds_ref = ds_ref.sel(time=slice(params['start_date'], params['end_date']))
    ref_da = ds_ref[ref_cfg['variable_name']]

    # === Compute Metric for all Datasets ===
    datasets_to_evaluate = [ref_cfg] + datasets_cfg
    
    for dat in datasets_to_evaluate:
        print(f"\nProcessing {dat['name']}...")
        
        if dat['name'] == ref_cfg['name']:
            da = ref_da
        else:
            ds_mod = xr.open_dataset(os.path.join(POST_PROCESS_ROOT, dat['file_path']))
            ds_mod = utils.mask_dataset(ds_mod, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
            ds_mod = ds_mod.sel(time=slice(params['start_date'], params['end_date']))
            if ds_mod.time.size == 0:
                print(f"Warning: {dat['name']} has no data in range {params['start_date']} - {params['end_date']}. Skipping.")
                continue
            da = ds_mod[dat['variable_name']]

        # Compute R01 for all seasons
        seasonal_maps, df = utils.compute_generic_metrics(
            da, None, compute_r01_frequency, 
            model_name=dat['name'], 
            suffix=general_suffix,
            output_dir=DATA_RESULTS_DIR,
            strategy=STRATEGY,
            corr_strategy="per_year",
            return_by_year=True,
            threshold=THRESHOLD
        )
        
        csv_path = os.path.join(DATA_RESULTS_DIR, f"seasonal_metrics_{dat['name']}_{general_suffix}_{STRATEGY}.csv")
        df.to_csv(csv_path)
        print(f"Saved seasonal means to: {csv_path}")

    # --- Plotting Loop ---
    nrows = 1
    plot_periods = params.get('plot_periods', ['Annual'])
    
    # Base paths for loading
    MODEL_PATHS_BASE = {
        d['name'].upper(): os.path.join(DATA_RESULTS_DIR, f"{d['name']}_{general_suffix}_strategy_{STRATEGY}_corr_per_year_PERIOD.nc")
        for d in datasets_to_evaluate
    }

    for period in plot_periods:
        print(f"\nGenerating plots for period: {period}...")
        
        dir_name = "all" if period == "Annual" else period
        period_fig_dir = os.path.join(RESULTS_DIR, "figures", dir_name)
        os.makedirs(period_fig_dir, exist_ok=True)

        data_raw = utils.load_metric_results(MODEL_PATHS_BASE, period=period)
        if not data_raw:
            continue
            
        data_spatial = {name: ds.mean("year") if "year" in ds.dims else ds for name, ds in data_raw.items()}

        # 1. Spatial Map
        spatial_path = os.path.join(period_fig_dir, f"{METRIC}_spatial_map_{period}.png")
        utils.plot_spatial_maps(data_spatial, METRIC, period=period, save_path=spatial_path, unit="%", nrows=nrows)

        # 2. Temporal Evolution
        evol_path = os.path.join(period_fig_dir, f"{METRIC}_evolution_{period}.png")
        utils.plot_temporal_evolution(MODEL_PATHS_BASE, METRIC, period=period, save_path=evol_path, unit="%")
        
        # 3. Boxplot
        box_path = os.path.join(period_fig_dir, f"{METRIC}_boxplot_{period}.png")
        utils.plot_metric_boxplot(MODEL_PATHS_BASE, METRIC, period=period, save_path=box_path)

    print(f"\n{METRIC.upper()} Metric completed successfully! ✅")

if __name__ == "__main__":
    main()
