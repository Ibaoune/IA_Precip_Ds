# Author: M. El Aabaribaoune (@um6p)
import os
import sys
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Add project root to sys.path
root_path = str(Path(__file__).resolve().parents[3])
if root_path not in sys.path:
    sys.path.append(root_path)

import utils

warnings.filterwarnings('ignore')

def compute_rocss(dataset, ref, threshold=1.0, return_by_year=False, **kwargs):
    """
    Computes ROC Skill Score (ROCSS) = 2 * AUC - 1.
    Evaluates discrimination of rain events (>= threshold).
    """
    # Binary event from reference
    y_true_full = (ref >= threshold).astype(int)
    y_score_full = dataset
    
    def _auc_wrapper(t, s):
        # Helper for vectorized calculation
        mask = ~np.isnan(t) & ~np.isnan(s)
        t, s = t[mask], s[mask]
        if len(t) == 0 or len(np.unique(t)) <= 1:
            return np.nan
        try:
            return roc_auc_score(t, s)
        except:
            return np.nan

    def _calc_rocss_map(score_da, true_da):
        # Vectorize using apply_ufunc across the time dimension
        auc_map = xr.apply_ufunc(
            _auc_wrapper,
            true_da,
            score_da,
            input_core_dims=[['time'], ['time']],
            vectorize=True,
            output_dtypes=[float],
            dask="parallelized"
        )
        rocss_map = 2 * auc_map - 1
        return rocss_map.transpose('lat', 'lon')

    if return_by_year:
        years = np.unique(dataset['time.year'].values)
        results = []
        for yr in years:
            s_yr = y_score_full.sel(time=str(yr))
            t_yr = y_true_full.sel(time=str(yr))
            res = _calc_rocss_map(s_yr, t_yr)
            results.append(res.assign_coords(year=yr))
        return xr.concat(results, dim='year')
    else:
        return _calc_rocss_map(y_score_full, y_true_full)

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
    
    reference_info = {
        "name": ref_cfg['name'],
        "file_path": os.path.join(POST_PROCESS_ROOT, ref_cfg['file_path']),
        "variable_name": ref_cfg['variable_name']
    }

    region_name = params['region']
    mask_type = "land" if params['mask_land'] else "whole"
    general_suffix = f"{params['predictand']}_{region_name}_calcul_{mask_type}"

    print(f"[DEBUG] Processing reference data for ROCSS...")
    ds_ref = xr.open_dataset(reference_info["file_path"])
    ds_ref = utils.mask_dataset(ds_ref, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
    ds_ref = ds_ref.sel(time=slice(params['start_date'], params['end_date']))
    var_ref = ds_ref[reference_info["variable_name"]]

    threshold = metric_cfg.get('threshold', 1.0)
    strategy = metric_cfg.get('strategy', 'mean_first')
    
    for dat in datasets_cfg:
        print(f"\nProcessing dataset: {dat['name']}")
        ds_mod = xr.open_dataset(os.path.join(POST_PROCESS_ROOT, dat["file_path"]))
        ds_mod = utils.mask_dataset(ds_mod, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
        ds_mod = ds_mod.sel(time=slice(params['start_date'], params['end_date']))
        if ds_mod.time.size == 0:
            print(f"Warning: {dat['name']} has no data in range {params['start_date']} - {params['end_date']}. Skipping.")
            continue
        var_mod = ds_mod[dat["variable_name"]]
        
        # We use compute_generic_metrics to handle seasonality
        _, df = utils.compute_generic_metrics(
            dataset=var_mod,
            ref=var_ref,
            metric_func=compute_rocss,
            model_name=dat["name"],
            suffix=general_suffix,
            output_dir=DATA_RESULTS_DIR,
            strategy=strategy,
            corr_strategy="per_year",
            return_by_year=True,
            threshold=threshold
        )
        
        csv_path = os.path.join(DATA_RESULTS_DIR, f"seasonal_metrics_{dat['name']}_{general_suffix}.csv")
        df.to_csv(csv_path)

    print("\nROCSS Calculation completed! ")

if __name__ == "__main__":
    main()
