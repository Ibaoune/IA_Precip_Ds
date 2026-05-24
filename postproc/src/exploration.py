import os
import sys
import warnings
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to sys.path
root_path = str(Path(__file__).resolve().parents[1])
src_path = os.path.join(root_path, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
if root_path not in sys.path:
    sys.path.append(root_path)

import utils

warnings.filterwarnings('ignore')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_explore.yaml", help="Path to exploration config file")
    args, unknown = parser.parse_known_args()

    # === Load Configuration ===
    config_path = args.config if os.path.isabs(args.config) else os.path.join(os.path.dirname(__file__), args.config)
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please create a config_explore.yaml or provide a valid path.")
        return

    config = utils.load_config(config_path)
    
    params = config['parameters']
    ref_cfg = config['reference']
    datasets_cfg = config['datasets']
    
    POST_PROCESS_ROOT = root_path
    
    # === Output Path Setup ===
    exp_name = config.get('experiment', 'exploration')
    RESULTS_DIR = os.path.join(POST_PROCESS_ROOT, "results", exp_name, "explore")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    region_name = params['region']
    
    # === Load Datasets ===
    all_datasets = {}
    
    # 1. Reference
    print(f"Loading reference: {ref_cfg['name']}...")
    ds_ref = xr.open_dataset(os.path.join(POST_PROCESS_ROOT, ref_cfg['file_path']))
    ds_ref = utils.mask_dataset(ds_ref, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
    ds_ref = ds_ref.sel(time=slice(params['start_date'], params['end_date']))
    all_datasets[ref_cfg['name'].upper()] = ds_ref[ref_cfg['variable_name']]
    
    # 2. Models
    for d in datasets_cfg:
        print(f"Loading model: {d['name']}...")
        fpath = os.path.join(POST_PROCESS_ROOT, d['file_path'])
        if not os.path.exists(fpath):
            print(f"Warning: File not found {fpath}. Skipping.")
            continue
            
        ds_mod = xr.open_dataset(fpath)
        ds_mod = utils.mask_dataset(ds_mod, region=region_name, only_land=params['mask_land'], only_morocco=params.get('only_morocco', True))
        ds_mod = ds_mod.sel(time=slice(params['start_date'], params['end_date']))
        
        if ds_mod.time.size == 0:
            print(f"Warning: {d['name']} has no overlap with period {params['start_date']} to {params['end_date']}. Skipping.")
            continue
            
        if d['variable_name'] not in ds_mod:
            print(f"Warning: Variable '{d['variable_name']}' not found in {d['name']}. Skipping.")
            continue

        all_datasets[d['name'].upper()] = ds_mod[d['variable_name']]

    # === Generate Exploration Plots ===
    print("\n--- Generating Exploration Plots ---")
    
    # 1. Monthly Annual Cycle
    print("1. Plotting Mean Annual Cycle...")
    cycle_path = os.path.join(RESULTS_DIR, "monthly_annual_cycle.png")
    utils.plot_monthly_cycle(all_datasets, region=region_name, save_path=cycle_path)
    
    # 2. Intensity Distribution (PDF)
    print("2. Plotting Intensity Distributions (PDF)...")
    
    # a. Log-Log Plot (Extremes)
    pdf_log_path = os.path.join(RESULTS_DIR, "intensity_pdf_log.png")
    utils.plot_intensity_distribution_log(all_datasets, region=region_name, save_path=pdf_log_path)
    
    # b. Linear KDE Plot (Bulk distribution - matches pdf.py style)
    pdf_lin_path = os.path.join(RESULTS_DIR, "intensity_pdf_linear.png")
    utils.plot_intensity_distribution_linear(all_datasets, region=region_name, save_path=pdf_lin_path)
    
    # 3. Spatial Climatology (Mean of whole period)
    print("3. Plotting Spatial Climatology Maps...")
    spatial_means = {name: ds.mean(dim='time') for name, ds in all_datasets.items()}
    spatial_path = os.path.join(RESULTS_DIR, "spatial_climatology.png")
    utils.plot_spatial_maps(spatial_means, "mean", period="Full Period", 
                               save_path=spatial_path, 
                               title=f"Spatial Climatology ({params['start_date']} - {params['end_date']})",
                               nrows=1)
    
    # 4. Daily Domain Mean Evolution
    print("4. Plotting Daily Evolution (Full Period)...")
    evol_path = os.path.join(RESULTS_DIR, "daily_evolution.png")
    
    # We create a dummy paths dict for the evolution function
    # But wait, plot_temporal_evolution loads from disk. 
    # To avoid re-loading, we can just implement it here or mock the paths.
    # Actually, let's just make it simple here.
    plt.figure(figsize=(12, 6), dpi=300)
    for i, (name, ds) in enumerate(all_datasets.items()):
        # Calculate domain mean daily
        daily_mean = ds.mean(dim=['lat', 'lon'])
        plt.plot(daily_mean.time, daily_mean.values, label=name, alpha=0.7)
    
    plt.title(f"Daily Domain Mean Precipitation ({region_name})", fontsize=14, fontweight='bold')
    plt.ylabel("Precipitation (mm/day)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(evol_path, dpi=400)
    plt.close()
    print(f"Saved daily evolution: {evol_path}")

    print(f"\nExploration completed! Results saved in: {RESULTS_DIR} ✅")

if __name__ == "__main__":
    main()
