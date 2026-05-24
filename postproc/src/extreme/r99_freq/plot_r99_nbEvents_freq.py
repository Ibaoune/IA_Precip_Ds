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
    STRATEGY = metric_cfg.get('strategy', 'daily_first')
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
    
    ALL_PATHS = MODEL_PATHS_BASE

    plot_periods = params.get('plot_periods', ['Annual'])
    nrows = 1

    for period in plot_periods:
        print(f"\nProcessing plots for period: {period}")
        
        dir_name = "all" if period == "Annual" else period
        period_fig_dir = os.path.join(RESULTS_DIR, "figures", dir_name)
        os.makedirs(period_fig_dir, exist_ok=True)

        data_raw = utils.load_metric_results(ALL_PATHS, period=period)
        if not data_raw:
            print(f"Skipping {period} - no data found.")
            continue

        data_spatial = {name: ds.mean("year") if "year" in ds.dims else ds for name, ds in data_raw.items()}

        # Reference Map
        ref_name = ref_cfg['name'].upper()
        ref_map = data_spatial.get(ref_name)

        # Plot 1: Spatial Map
        spatial_path = os.path.join(period_fig_dir, f"{METRIC}_spatial_map_{period}.png")
        print(f"Generating spatial comparison map for {period}...")
        utils.plot_spatial_maps(data_spatial, "r95p", period=period, save_path=spatial_path, unit="days", nrows=nrows)

        # Plot 2: Spatial Error Map
        if ref_map is not None:
            error_path = os.path.join(period_fig_dir, f"{METRIC}_spatial_error_{period}.png")
            print(f"Generating spatial error map for {period}...")
            error_dict = {name: ds - ref_map for name, ds in data_spatial.items() if name != ref_name}
            if error_dict:
                utils.plot_spatial_maps(error_dict, "frequency_error", period=period, save_path=error_path, unit="days", title=f"{period} R99 Freq Error", nrows=nrows)

        # Plot 3: Temporal Evolution
        evol_path = os.path.join(period_fig_dir, f"{METRIC}_evolution_{period}.png")
        print(f"Generating temporal evolution plot for {period}...")
        utils.plot_temporal_evolution(ALL_PATHS, "r95p", period=period, save_path=evol_path, unit="days")
        
        # Plot 4: Boxplot
        box_path = os.path.join(period_fig_dir, f"{METRIC}_boxplot_{period}.png")
        print(f"Generating distribution boxplots for {period}...")
        utils.plot_metric_boxplot(ALL_PATHS, "r95p", period=period, save_path=box_path)
        
    print("\nPlotting completed successfully! ✅")

if __name__ == "__main__":
    main()
