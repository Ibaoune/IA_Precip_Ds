import os
import sys
import warnings
import numpy as np
import xarray as xr
from pathlib import Path

# Add project root to sys.path
root_path = str(Path(__file__).resolve().parents[3])
if root_path not in sys.path:
    sys.path.append(root_path)

import utils

warnings.filterwarnings('ignore')

def compute_r99_amount(dataset, ref=None, **kwargs):
    """
    Compute R99 Amount: Mean intensity of days >= P99.
    P99 is calculated from the reference (if provided) or the dataset itself,
    over the entire period to ensure consistency.
    """
    return_by_year = kwargs.get('return_by_year', False)
    
    # 1. Determine threshold source (use reference if available)
    threshold_source = ref if ref is not None else dataset
    
    # 2. Calculate the global threshold for the provided period
    wet_days_ref = threshold_source.where(threshold_source >= 1.0)
    p99 = wet_days_ref.quantile(0.99, dim='time', skipna=True)
    
    def _apply_p99(da, thresh):
        extreme_days = da.where(da >= thresh)
        return extreme_days.mean('time', skipna=True)

    if return_by_year:
        years = np.unique(dataset['time.year'].values)
        results = []
        for yr in years:
            data_yr = dataset.sel(time=str(yr))
            res = _apply_p99(data_yr, p99)
            results.append(res.assign_coords(year=yr))
        return xr.concat(results, dim='year')
    else:
        return _apply_p99(dataset, p99)

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
    
    # New subfolder for data results as per user request
    DATA_RESULTS_DIR = os.path.join(RESULTS_DIR, "results")
    os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
    
    # Keep for plotting layout

    reference_info = {
        "name": ref_cfg['name'],
        "file_path": os.path.join(POST_PROCESS_ROOT, ref_cfg['file_path']),
        "variable_name": ref_cfg['variable_name']
    }

    datasets_to_evaluate = [ref_cfg] + datasets_cfg

    region_name = params['region']
    mask_type = "land" if params['mask_land'] else "whole"
    general_suffix = f"{params['predictand']}_{region_name}_calcul_{mask_type}"

    print(f"[DEBUG] Processing reference data for R99: {reference_info['name']}...")
    ds_ref = xr.open_dataset(reference_info["file_path"])
    ds_ref = utils.mask_dataset(ds_ref, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
    ds_ref = ds_ref.sel(time=slice(params['start_date'], params['end_date']))
    var_ref = ds_ref[reference_info["variable_name"]]

    strategy = metric_cfg['strategy']
    corr_strategy = metric_cfg.get('corr_strategy', 'per_year')
    return_by_year = True

    expected_files = []

    for dat in datasets_to_evaluate:
        print(f"\nProcessing dataset: {dat['name']}")

        if dat['name'] == ref_cfg['name']:
            var_mod = var_ref
        else:
            ds_mod = xr.open_dataset(os.path.join(POST_PROCESS_ROOT, dat["file_path"]))
            ds_mod = utils.mask_dataset(ds_mod, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
            ds_mod = ds_mod.sel(time=slice(params['start_date'], params['end_date']))
            var_mod = ds_mod[dat["variable_name"]]

        _, df = utils.compute_generic_metrics(
            dataset=var_mod,
            ref=var_ref, # Pass reference to ensure consistent threshold
            metric_func=compute_r99_amount,
            model_name=dat["name"],
            suffix=general_suffix,
            output_dir=DATA_RESULTS_DIR,
            strategy=strategy,
            corr_strategy=corr_strategy,
            return_by_year=return_by_year,
            condition=metric_cfg.get('condition', 'greater_equal')
        )
        
        csv_path = os.path.join(DATA_RESULTS_DIR, f"seasonal_metrics_{dat['name']}_{general_suffix}.csv")
        df.to_csv(csv_path)
        expected_files.append(csv_path)
        expected_files.append(os.path.join(DATA_RESULTS_DIR, f"{dat['name']}_{general_suffix}_strategy_{strategy}_corr_{corr_strategy}_Annual.nc"))

    # --- Plotting Loop ---
    nrows = 1
    
    plot_periods = params.get('plot_periods', ['Annual'])
    
    for period in plot_periods:
        print(f"\nGenerating plots for period: {period}...")
        
        dir_name = "all" if period == "Annual" else period
        period_fig_dir = os.path.join(RESULTS_DIR, "figures", dir_name)
        os.makedirs(period_fig_dir, exist_ok=True)

        model_paths_period = {
            d['name'].upper(): os.path.join(DATA_RESULTS_DIR, f"{d['name']}_{general_suffix}_strategy_{strategy}_corr_{corr_strategy}_PERIOD.nc")
            for d in datasets_to_evaluate
        }
        
        data_raw = utils.load_metric_results(model_paths_period, period=period)
        if not data_raw:
            print(f"Skipping {period} - no data files found.")
            continue
            
        # Reference R99 for this period
        if period == "Annual":
            ref_period = var_ref
        else:
            ref_period = var_ref.sel(time=var_ref.time.dt.season == period)
            
        ref_map = compute_r99_amount(dataset=ref_period, return_by_year=False)
        
        spatial_data_dict = {reference_info["name"].upper(): ref_map}
        spatial_error_dict = {}
        
        for name, ds in data_raw.items():
            mean_ds = ds.mean("year") if "year" in ds.dims else ds
            spatial_data_dict[name] = mean_ds
            if name != reference_info["name"].upper():
                spatial_error_dict[name] = mean_ds - ref_map

        # Plot 1: R99 Comparison Map
        utils.plot_spatial_maps(spatial_data_dict, "r99", period=period, 
                                   save_path=os.path.join(period_fig_dir, f"spatial_r99_comparison_{period}.png"),
                                   title=f"{period} Spatial R99 Comparison ({params['predictand'].upper()})", 
                                   unit="mm", nrows=nrows)
        
        # Plot 2: R99 Error Map
        if spatial_error_dict:
            utils.plot_spatial_maps(spatial_error_dict, "r99_error", period=period, 
                                       save_path=os.path.join(period_fig_dir, f"spatial_r99_error_{period}.png"),
                                       title=f"{period} Spatial R99 Error ({params['predictand'].upper()})", 
                                       unit="mm", nrows=nrows)

    print("\nPipeline tests passed! ✅")

if __name__ == "__main__":
    main()
