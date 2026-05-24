# Author: M. El Aabaribaoune (@um6p)
import os
import sys
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# Add project root to sys.path
root_path = str(Path(__file__).resolve().parents[3])
if root_path not in sys.path:
 sys.path.append(root_path)

import utils

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
 datasets_cfg = config['datasets']
 ref_cfg = config['reference']

 METRIC = metric_cfg['name']
 STRATEGY = metric_cfg['strategy']
 CORR_STRATEGY = metric_cfg.get('corr_strategy', 'per_year')
 
 POST_PROCESS_ROOT = root_path
 
 # === Output Path Setup ===
 RESULTS_DIR = utils.get_results_dir(config, METRIC, POST_PROCESS_ROOT)
 DATA_RESULTS_DIR = os.path.join(RESULTS_DIR, "results")
 
 region_name = params['region']
 mask_type = "land" if params['mask_land'] else "whole"
 suffix = f"{params['predictand']}_{region_name}_calcul_{mask_type}"

 datasets_to_evaluate = [ref_cfg] + datasets_cfg

 # Build model paths dynamically
 MODEL_PATHS_BASE = {
 d['name'].upper(): os.path.join(DATA_RESULTS_DIR, f"{d['name']}_{suffix}_strategy_{STRATEGY}_corr_{CORR_STRATEGY}_PERIOD.nc")
 for d in datasets_to_evaluate
 }

 plot_periods = params.get('plot_periods', ['Annual'])
 
 for period in plot_periods:
 print(f"\nProcessing plots for period: {period}")
 
 dir_name = "all" if period == "Annual" else period
 period_fig_dir = os.path.join(RESULTS_DIR, "figures", dir_name)
 os.makedirs(period_fig_dir, exist_ok=True)

 data_raw = utils.load_metric_results(MODEL_PATHS_BASE, period=period)
 if not data_raw:
 print(f"Skipping {period} - no data found in {DATA_RESULTS_DIR}.")
 continue

 data_spatial = {name: ds.mean("year") if "year" in ds.dims else ds for name, ds in data_raw.items()}

 # Plot 1: Spatial Map
 spatial_path = os.path.join(period_fig_dir, f"{METRIC}_spatial_map_{period}.png")
 print(f"Generating spatial map for {period}...")
 utils.plot_spatial_maps(data_spatial, METRIC, period=period, save_path=spatial_path, unit="mm")

 # Plot 2: Temporal Evolution
 evol_path = os.path.join(period_fig_dir, f"{METRIC}_evolution_{period}.png")
 print(f"Generating temporal evolution plot for {period}...")
 utils.plot_temporal_evolution(MODEL_PATHS_BASE, METRIC, period=period, save_path=evol_path, unit="mm")
 
 # Plot 3: Boxplots
 box_path = os.path.join(period_fig_dir, f"{METRIC}_boxplot_{period}.png")
 print(f"Generating distribution boxplots for {period}...")
 utils.plot_metric_boxplot(MODEL_PATHS_BASE, METRIC, period=period, save_path=box_path)

 print("\nPlotting completed successfully! ")

if __name__ == "__main__":
 main()
