import os
import sys
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.stats import pearsonr

# Add project root to sys.path to allow importing utils
root_path = str(Path(__file__).resolve().parents[3])
if root_path not in sys.path:
    sys.path.append(root_path)

import utils

warnings.filterwarnings('ignore')

def compute_corr_flagged(dataset, ref, strategy='per_year', return_by_year=False, use_anomaly=True):
    """
    Standard correlation calculation.
    """
    dataset, ref = xr.align(dataset, ref, join='inner')

    if strategy == 'per_year':
        def _pearson_wrapper(d_yr, r_yr):
            d_flat = d_yr.ravel()
            r_flat = r_yr.ravel()
            mask = ~np.isnan(d_flat) & ~np.isnan(r_flat)
            if np.sum(mask) < 2 or np.std(d_flat[mask]) == 0 or np.std(r_flat[mask]) == 0:
                return np.nan
            return pearsonr(d_flat[mask], r_flat[mask])[0]

        years = np.unique(dataset['time.year'].values)
        corr_list = []
        for yr in years:
            d_yr = dataset.sel(time=str(yr))
            r_yr = ref.sel(time=str(yr))
            
            if use_anomaly:
                d_yr = d_yr - d_yr.mean('time')
                r_yr = r_yr - r_yr.mean('time')
            
            corr_map = xr.apply_ufunc(
                _pearson_wrapper, d_yr, r_yr,
                input_core_dims=[['time'], ['time']],
                vectorize=True,
                dask='parallelized',
                output_dtypes=[float]
            )
            corr_list.append(corr_map.assign_coords(year=yr))
        
        full_corr = xr.concat(corr_list, dim='year')
        return full_corr if return_by_year else full_corr.mean('year')
        
    elif strategy == 'daily_first':
        if use_anomaly:
            dataset = dataset.groupby('time.month') - dataset.groupby('time.month').mean('time')
            ref = ref.groupby('time.month') - ref.groupby('time.month').mean('time')
            
        def _corr_vec(d, r):
            mask = ~np.isnan(d) & ~np.isnan(r)
            if np.sum(mask) < 2 or np.std(d[mask]) == 0 or np.std(r[mask]) == 0:
                return np.nan
            return pearsonr(d[mask], r[mask])[0]

        return xr.apply_ufunc(
            _corr_vec, dataset.drop_vars('month') if 'month' in dataset.coords else dataset, 
            ref.drop_vars('month') if 'month' in ref.coords else ref,
            input_core_dims=[['time'], ['time']],
            vectorize=True,
            output_dtypes=[float]
        )
    else:
        raise ValueError("Strategy not supported for correlation")

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
    
    # New subfolder for data results
    DATA_RESULTS_DIR = os.path.join(RESULTS_DIR, "results")
    os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
    
    # Keep for plotting layout

    # Reference
    reference = {
        "name": ref_cfg['name'],
        "file_path": os.path.join(POST_PROCESS_ROOT, ref_cfg['file_path']),
        "variable_name": ref_cfg['variable_name']
    }

    # Datasets to evaluate
    datasets_to_evaluate = []
    for d in datasets_cfg:
        datasets_to_evaluate.append({
            "name": d['name'],
            "file_path": os.path.join(POST_PROCESS_ROOT, d['file_path']),
            "variable_name": d['variable_name']
        })

    region_name = params['region']
    mask_type = "land" if params['mask_land'] else "whole"
    general_suffix = f"{params['predictand']}_{region_name}_calcul_{mask_type}"

    print(f"[DEBUG] Processing reference data for Correlation: {reference['name']}...")
    ds_ref = xr.open_dataset(reference["file_path"])
    ds_ref = utils.mask_dataset(ds_ref, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
    ds_ref = ds_ref.sel(time=slice(params['start_date'], params['end_date']))
    var_ref = ds_ref[reference["variable_name"]]

    strategy = metric_cfg['strategy']
    corr_strategy = metric_cfg['corr_strategy']
    return_by_year = True 
    use_anomaly = True 

    expected_files = []

    for dat in datasets_to_evaluate:
        print(f"\nProcessing dataset: {dat['name']}")
        
        ds_mod = xr.open_dataset(dat["file_path"])
        ds_mod = utils.mask_dataset(ds_mod, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
        ds_mod = ds_mod.sel(time=slice(params['start_date'], params['end_date']))
        var_mod = ds_mod[dat["variable_name"]]

        _, df = utils.compute_generic_metrics(
            dataset=var_mod,
            ref=var_ref,
            metric_func=compute_corr_flagged,
            model_name=dat["name"],
            suffix=general_suffix,
            output_dir=DATA_RESULTS_DIR,
            strategy=strategy,
            corr_strategy=corr_strategy,
            return_by_year=return_by_year,
            use_anomaly=use_anomaly
        )
        
        csv_path = os.path.join(DATA_RESULTS_DIR, f"seasonal_metrics_{dat['name']}_{general_suffix}_{strategy}.csv")
        df.to_csv(csv_path)
        print(f"Results saved: {csv_path}")

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
            
        spatial_corr_dict = {}
        for name, ds in data_raw.items():
            mean_ds = ds.mean("year") if "year" in ds.dims else ds
            spatial_corr_dict[name] = mean_ds

        # Plot 1: Correlation Map
        utils.plot_spatial_maps(spatial_corr_dict, "correlation", period=period, 
                                   save_path=os.path.join(period_fig_dir, f"spatial_correlation_{period}.png"),
                                   title=f"{period} Spatial Correlation ({params['predictand'].upper()})", 
                                   unit="r", nrows=nrows)
        
        # Plot 2: Boxplot
        box_path = os.path.join(period_fig_dir, f"correlation_boxplot_{period}.png")
        utils.plot_metric_boxplot(model_paths_period, "correlation", period=period, save_path=box_path)

    if "Annual" in plot_periods:
        evol_path = os.path.join(RESULTS_DIR, "figures", "all", "correlation_evolution_Annual.png")
        model_paths_annual = {
            d['name'].upper(): os.path.join(DATA_RESULTS_DIR, f"{d['name']}_{general_suffix}_strategy_{strategy}_corr_{corr_strategy}_Annual.nc")
            for d in datasets_to_evaluate
        }
        utils.plot_temporal_evolution(model_paths_annual, "correlation", period="Annual", save_path=evol_path, unit="r")
        
    print("\n=== PIPELINE TESTS ===")
    all_passed = all(os.path.exists(f) for f in expected_files)
    for f in expected_files:
        status = "PASSED" if os.path.exists(f) else "FAILED"
        print(f"[{status}]: File created -> {os.path.basename(f)}")
            
    assert all_passed, "Pipeline failed: Not all expected result files were generated."
    print("All Pipeline tests passed successfully! ✅")

if __name__ == "__main__":
    main()
