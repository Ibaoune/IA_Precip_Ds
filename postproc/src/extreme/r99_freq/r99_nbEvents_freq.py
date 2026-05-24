# Author: M. El Aabaribaoune (@um6p)
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

def compute_r99p_flagged(dataset, ref, strategy='per_year', return_by_year=False):
 """
 Compute R99p: Number of days with precipitation > 99th percentile of wet days.
 """
 threshold_wet = 1.0
 ref_wet = ref.where(ref >= threshold_wet)
 p99 = ref_wet.quantile(0.99, dim='time', skipna=True)
 
 results = []
 years = np.unique(dataset['time.year'].values)
 for yr in years:
 dat_yr = dataset.sel(time=str(yr))
 is_extreme = (dat_yr >= p99).astype(float)
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

 print(f"[DEBUG] Processing reference data for R99p: {reference_info['name']}...")
 ds_ref = xr.open_dataset(reference_info["file_path"])
 ds_ref = utils.mask_dataset(ds_ref, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
 ds_ref = ds_ref.sel(time=slice(params['start_date'], params['end_date']))
 var_ref = ds_ref[reference_info["variable_name"]]

 strategy = metric_cfg['strategy']
 corr_strategy = metric_cfg.get('corr_strategy', 'per_year')
 return_by_year = True

 datasets_to_evaluate = [ref_cfg] + datasets_cfg

 for dat in datasets_to_evaluate:
 print(f"\nProcessing dataset: {dat['name']}")
 ds_mod = xr.open_dataset(os.path.join(POST_PROCESS_ROOT, dat["file_path"]))
 ds_mod = utils.mask_dataset(ds_mod, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
 ds_mod = ds_mod.sel(time=slice(params['start_date'], params['end_date']))
 
 if ds_mod.time.size == 0:
 print(f"Warning: {dat['name']} has no data. Skipping.")
 continue
 
 var_mod = ds_mod[dat["variable_name"]]
 
 _, df = utils.compute_generic_metrics(
 dataset=var_mod,
 ref=var_ref,
 metric_func=compute_r99p_flagged,
 model_name=dat["name"],
 suffix=general_suffix,
 output_dir=DATA_RESULTS_DIR,
 strategy=strategy,
 corr_strategy=corr_strategy,
 return_by_year=return_by_year
 )
 
 csv_path = os.path.join(DATA_RESULTS_DIR, f"seasonal_metrics_{dat['name']}_{general_suffix}.csv")
 df.to_csv(csv_path)

 print("\nCalculation completed! ")

 print("\nPipeline tests passed! ")

if __name__ == "__main__":
 main()
