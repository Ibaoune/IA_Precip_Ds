# Author: M. El Aabaribaoune (@um6p)
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Add project root to sys.path
root_path = str(Path(__file__).resolve().parents[3])
if root_path not in sys.path:
 sys.path.append(root_path)

import utils

warnings.filterwarnings('ignore')

def plot_qq(df, ref_name, model_names, save_path, title=None, global_max_val=None):
 """
 Step 4: The Mapping.
 Creates a premium QQ-plot with 1:1 line and log scales if needed.
 """
 plt.figure(figsize=(8, 8), dpi=300)
 
 # Premium colors
 colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#8E44AD']
 
 ref_vals = df[ref_name].values
 
 # Determine max value for 1:1 line
 local_max = max(ref_vals.max(), max([df[m].max() for m in model_names]))
 max_val = global_max_val if global_max_val is not None else local_max
 
 # 1:1 Reference Line
 plt.plot([0, max_val], [0, max_val], color='black', linestyle='--', alpha=0.6, label='1:1 Line', linewidth=1.5)
 
 for i, model_name in enumerate(model_names):
 model_vals = df[model_name].values
 color = utils.GLOBAL_MODEL_COLORS.get(model_name.upper(), colors[i % len(colors)])
 plt.plot(ref_vals, model_vals, label=model_name, color=color, linewidth=2.5, alpha=0.9)
 
 plt.xlabel(f"Observations: {ref_name.upper()} (mm/day)", fontsize=12, fontweight='bold')
 plt.ylabel(f"Models (mm/day)", fontsize=12, fontweight='bold')
 
 if title:
 plt.title(title, fontsize=15, fontweight='bold', pad=20)
 else:
 plt.title("Quantile-Quantile Plot: Physical Integrity Check", fontsize=15, fontweight='bold', pad=20)
 
 plt.legend(frameon=True, fontsize=10, loc='upper left')
 plt.grid(True, linestyle=':', alpha=0.6)
 
 # Premium look: square plot
 plt.gca().set_aspect('equal', adjustable='box')
 plt.xlim(0, max_val)
 plt.ylim(0, max_val)
 
 # Add a log-log inset if there's high intensity data? 
 # Or just use log scales if requested. For now, linear is standard for "physical integrity" unless extremes are very high.
 
 plt.tight_layout()
 
 if save_path:
 os.makedirs(os.path.dirname(save_path), exist_ok=True)
 plt.savefig(save_path, dpi=500, bbox_inches='tight')
 plt.close()
 print(f"Saved QQ-plot: {save_path}")
 else:
 plt.show()

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
 DATA_RESULTS_DIR = os.path.join(RESULTS_DIR, "results")
 mask_type = "land" if params['mask_land'] else "whole"
 model_names = [d['name'] for d in datasets_cfg]
 plot_periods = params.get('plot_periods', ['Annual'])
 
 for period in plot_periods:
 print(f"\nGenerating QQ-plot for period: {period}...")
 
 dir_name = "all" if period == "Annual" else period
 period_fig_dir = os.path.join(RESULTS_DIR, "figures", dir_name)
 os.makedirs(period_fig_dir, exist_ok=True)
 
 csv_filename = f"qqplot_data_{params['predictand']}_{params['region']}_{mask_type}_{period}.csv"
 csv_path = os.path.join(DATA_RESULTS_DIR, csv_filename)
 
 if not os.path.exists(csv_path):
 print(f"Error: Data file not found at {csv_path}. Run qqplot.py first for this period.")
 continue
 
 df = pd.read_csv(csv_path)
 
 # Determine global max from allmorr if we are processing a subregion
 global_max_val = None
 if params['region'] != "allmorr":
 try:
 allmorr_csv_path = csv_path.replace(f"_{params['region']}_", "_allmorr_").replace(f"/{params['region']}/", "/allmorr/")
 if os.path.exists(allmorr_csv_path):
 df_allmorr = pd.read_csv(allmorr_csv_path)
 global_max_val = max(df_allmorr[ref_cfg['name']].max(), max([df_allmorr[m].max() for m in model_names]))
 except Exception as e:
 print(f"Warning: Could not infer allmorr limits for QQ-plot. {e}")

 save_path = os.path.join(period_fig_dir, f"qqplot_{params['predictand']}_{params['region']}_{period}.png")
 
 title_str = f"Quantile-Quantile Plot ({period})"
 title_str += utils.get_title_metadata('qqplot', period)
 
 plot_qq(df, ref_cfg['name'], model_names, save_path, title=title_str, global_max_val=global_max_val)

if __name__ == "__main__":
 main()
