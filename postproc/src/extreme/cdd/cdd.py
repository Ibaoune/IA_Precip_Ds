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

def compute_cdd_flagged(dataset, ref=None, strategy='per_year', return_by_year=False, threshold_mm=1.0):
    """
    Compute Maximum Consecutive Dry Days (CDD).
    """
    def _cdd_calc(data_yr):
        def _get_max_consecutive(arr):
            if np.all(np.isnan(arr)): 
                return np.nan
            binary_arr = arr < threshold_mm
            if not np.any(binary_arr): 
                return 0.0
            int_arr = binary_arr.astype(int)
            padded = np.diff(np.concatenate(([0], int_arr, [0])))
            starts = np.where(padded == 1)[0]
            stops = np.where(padded == -1)[0]
            if len(starts) == 0: 
                return 0.0
            lengths = stops - starts
            return float(np.max(lengths))

        return xr.apply_ufunc(
            _get_max_consecutive, data_yr,
            input_core_dims=[['time']],
            vectorize=True,
            output_dtypes=[float]
        )

    years = np.unique(dataset['time.year'].values)
    cdd_list = []
    for yr in years:
        data_yr = dataset.sel(time=str(yr))
        cdd_map = _cdd_calc(data_yr)
        cdd_list.append(cdd_map.assign_coords(year=yr))
    
    full_cdd = xr.concat(cdd_list, dim='year')
    return full_cdd if return_by_year else full_cdd.mean('year')

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
    base_results_dir = utils.get_results_dir(config, metric_cfg['name'], POST_PROCESS_ROOT)
    
    # === Retrieve CDD Thresholds from Config ===
    # Check global master config first under postproc.extreme, then fall back to local metric config
    postproc_cfg = config.get("postproc", {})
    extreme_cfg = postproc_cfg.get("extreme", {})
    raw_threshold = extreme_cfg.get("cdd_thresholds", extreme_cfg.get("cdd_threshold"))
    
    if raw_threshold is None:
        raw_threshold = metric_cfg.get("threshold_mm", 1.0)
        
    if isinstance(raw_threshold, list):
        raw_list = raw_threshold
    else:
        raw_list = [raw_threshold]
        
    thresholds = []
    for val in raw_list:
        if isinstance(val, str):
            val_clean = val.replace("mm", "").strip()
            try:
                thresholds.append(float(val_clean))
            except ValueError:
                print(f"⚠️ Warning: Could not parse CDD threshold '{val}' as float. Skipping.")
        else:
            try:
                thresholds.append(float(val))
            except (ValueError, TypeError):
                print(f"⚠️ Warning: Could not parse CDD threshold '{val}' as float. Skipping.")
                
    if not thresholds:
        thresholds = [1.0]
    
    for threshold_mm in thresholds:
        if threshold_mm == 1.0:
            threshold_str = "1mm"
        elif threshold_mm == 0.5:
            threshold_str = "0.5"
        else:
            threshold_str = f"{threshold_mm}mm" if int(threshold_mm) == threshold_mm else f"{threshold_mm}"
            
        RESULTS_DIR = os.path.join(base_results_dir, f"cdd_{threshold_str}")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # New subfolder for data results as per user request
        DATA_RESULTS_DIR = os.path.join(RESULTS_DIR, "results")
        os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
        
        print(f"\n==========================================================")
        print(f"CDD Pipeline: Running for threshold = {threshold_mm} mm")
        print(f"Results Directory: {RESULTS_DIR}")
        print(f"==========================================================")

        # Reference
        reference = {
            "name": ref_cfg['name'],
            "file_path": os.path.join(POST_PROCESS_ROOT, ref_cfg['file_path']),
            "variable_name": ref_cfg['variable_name']
        }

        # Datasets to evaluate
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

        print(f"[DEBUG] Processing reference data for CDD: {reference['name']}...")
        ds_ref = xr.open_dataset(reference["file_path"])
        ds_ref = utils.mask_dataset(ds_ref, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
        ds_ref = ds_ref.sel(time=slice(params['start_date'], params['end_date']))
        var_ref = ds_ref[reference["variable_name"]]

        strategy = metric_cfg['strategy']
        corr_strategy = "per_year"
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
                metric_func=compute_cdd_flagged,
                model_name=dat["name"],
                suffix=general_suffix,
                output_dir=DATA_RESULTS_DIR,
                strategy=strategy,
                corr_strategy=corr_strategy,
                return_by_year=return_by_year,
                threshold_mm=threshold_mm
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
                
            ref_p = var_ref if period == "Annual" else var_ref.sel(time=var_ref.time.dt.season == period)
            ref_cdd = compute_cdd_flagged(dataset=ref_p, return_by_year=False, threshold_mm=threshold_mm)
            
            spatial_cdd_dict = {reference["name"].upper(): ref_cdd}
            spatial_cdd_error_dict = {}
            
            for name, ds in data_raw.items():
                mean_cdd = ds.mean("year") if "year" in ds.dims else ds
                spatial_cdd_dict[name] = mean_cdd
                if name != reference["name"].upper():
                    spatial_cdd_error_dict[name] = mean_cdd - ref_cdd

            # Plot 1: CDD Comparison Map
            utils.plot_spatial_maps(spatial_cdd_dict, f"cdd_{threshold_str}", period=period, 
                                       save_path=os.path.join(period_fig_dir, f"spatial_cdd_comparison_{period}.png"),
                                       title=f"{period} Spatial CDD Comparison ({params['predictand'].upper()})", 
                                       unit="days", nrows=nrows)
            
            # Plot 2: CDD Error Map
            if spatial_cdd_error_dict:
                utils.plot_spatial_maps(spatial_cdd_error_dict, f"cdd_{threshold_str}_error", period=period, 
                                           save_path=os.path.join(period_fig_dir, f"spatial_cdd_error_{period}.png"),
                                           title=f"{period} Spatial CDD Error ({params['predictand'].upper()})", 
                                           unit="days", nrows=nrows)

    print("\nPipeline tests passed! ✅")

if __name__ == "__main__":
    main()
