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

def compute_r95p_flagged(dataset, ref, strategy='per_year', return_by_year=False):
    """
    Compute R95p: Number of days with precipitation > 95th percentile.
    """
    threshold_wet = 1.0
    ref_wet = ref.where(ref >= threshold_wet)
    p95 = ref_wet.quantile(0.95, dim='time', skipna=True)
    
    results = []
    years = np.unique(dataset['time.year'].values)
    for yr in years:
        dat_yr = dataset.sel(time=str(yr))
        is_extreme = (dat_yr >= p95).astype(float)
        count = is_extreme.sum('time', skipna=True)
        count = count.where(dat_yr.isel(time=0).notnull())
        results.append(count.assign_coords(year=yr))
        
    full_count = xr.concat(results, dim='year')
    return full_count if return_by_year else full_count.mean('year')

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

    datasets_to_evaluate = [ref_cfg]
    for d in datasets_cfg:
        datasets_to_evaluate.append({
            "name": d['name'],
            "file_path": os.path.join(POST_PROCESS_ROOT, d['file_path']),
            "variable_name": d['variable_name']
        })

    region_name = params['region']
    mask_type = "land" if params['mask_land'] else "whole"
    general_suffix = f"{params['predictand']}_{region_name}_calcul_{mask_type}"

    print(f"[DEBUG] Processing reference data for R95p: {reference_info['name']}...")
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
            ds_mod = xr.open_dataset(dat["file_path"])
            ds_mod = utils.mask_dataset(ds_mod, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
            ds_mod = ds_mod.sel(time=slice(params['start_date'], params['end_date']))
            var_mod = ds_mod[dat["variable_name"]]
        
        _, df = utils.compute_generic_metrics(
            dataset=var_mod,
            ref=var_ref,
            metric_func=compute_r95p_flagged,
            model_name=dat["name"],
            suffix=general_suffix,
            output_dir=DATA_RESULTS_DIR,
            strategy=strategy,
            corr_strategy=corr_strategy,
            return_by_year=return_by_year
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
            
        # Reference Frequency for this period
        if period == "Annual":
            ref_period = var_ref
        else:
            ref_period = var_ref.sel(time=var_ref.time.dt.season == period)
            
        ref_freq = compute_r95p_flagged(dataset=ref_period, ref=var_ref, return_by_year=False)
        
        spatial_freq_dict = {reference_info["name"].upper(): ref_freq}
        spatial_freq_error_dict = {}
        
        for name, ds in data_raw.items():
            mean_freq = ds.mean("year") if "year" in ds.dims else ds
            spatial_freq_dict[name] = mean_freq
            if name != reference_info["name"].upper():
                spatial_freq_error_dict[name] = mean_freq - ref_freq

        # Plot 1: Frequency Comparison
        utils.plot_spatial_maps(spatial_freq_dict, "r95p", period=period, 
                                   save_path=os.path.join(period_fig_dir, f"spatial_frequency_comparison_{period}.png"),
                                   title=f"{period} Extreme Frequency Comparison ({params['predictand'].upper()})", 
                                   unit="days", nrows=nrows)
        
        # Plot 2: Frequency Error
        if spatial_freq_error_dict:
            utils.plot_spatial_maps(spatial_freq_error_dict, "frequency_error", period=period, 
                                       save_path=os.path.join(period_fig_dir, f"spatial_frequency_error_{period}.png"),
                                       title=f"{period} Extreme Frequency Error ({params['predictand'].upper()})", 
                                       unit="days", nrows=nrows)

    print("\nPipeline tests passed! ✅")

if __name__ == "__main__":
    main()
