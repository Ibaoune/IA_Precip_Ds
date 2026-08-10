"""
Author: M. El Aabaribaoune (@um6p)
Description: Heatmap of skill change (Global → Regionalized-Loss) for mean precipitation
             metrics (RMSE, correlation, |Bias|) across sub-regions and seasons.

Adapted from plot_scenario2_added_value.py for the Retained_Regional_Evaluation experiment.

Mapping:
    Scenario1 (global)    →  {Arch}_Global
    Scenario2 (regional)  →  {Arch}_{Region}   (e.g. CNN_North, Unet_South, …)

Inputs (seasonal CSV files per sub-region produced by postproc.py + regional_summary.py):
    results/Retained_Regional_Evaluation/<dates>/<region>/test/bias/results/
        seasonal_metrics_{model}_pr_{region}_calcul_land_mean_first.csv
    results/Retained_Regional_Evaluation/<dates>/<region>/test/rmse/results/
        seasonal_metrics_{model}_pr_{region}_calcul_land_mean_first.csv
    results/Retained_Regional_Evaluation/<dates>/<region>/test/correlation/results/
        seasonal_metrics_{model}_pr_{region}_calcul_land_per_year.csv

Outputs (saved next to the results directory):
    results/Retained_Regional_Evaluation/<dates>/regional_added_value_heatmap_main.pdf/.png
    results/Retained_Regional_Evaluation/<dates>/regional_added_value_heatmap_supp.pdf/.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_RESULTS_DIR = "results/Retained_Regional_Evaluation/2006-01-01_2020-12-31"

# Sub-regions evaluated (must match the subfolders produced by postproc.py)
REGIONS = ['north', 'north_east', 'east', 'south']

# Seasons available in the CSV files
SEASONS = ['Annual', 'DJF', 'JJA']

# Architectures (base name, without _Global / _North / …)
ARCHITECTURES = ['Unet', 'CNN', 'Vit']   # note: 'ViT' → 'Vit' in file names

# Display names for the heatmap columns/labels
ARCH_DISPLAY = {'Unet': 'U-Net', 'CNN': 'CNN', 'Vit': 'ViT'}

# Display names for the sub-regions (left-hand labels)
REGION_DISPLAY = {'north': 'North', 'north_east': 'North-East', 'east': 'East', 'south': 'South'}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def extract_metric(path, col_name, season):
    import xarray as xr
    try:
        df = pd.read_csv(path, index_col=0)
        if season in df.index:
            return df.loc[season, col_name]
    except Exception:
        pass

    # Fallback to NetCDF file if the CSV does not have the season (e.g., Annual)
    try:
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path).replace("seasonal_metrics_", "").replace(".csv", "")
        parts = base_name.split("_pr_")
        model_part = parts[0]
        
        # Find the corresponding NetCDF file
        nc_files = [f for f in os.listdir(dir_name) if f.startswith(model_part + "_pr_") and f.endswith(f"_{season}.nc")]
        if nc_files:
            nc_path = os.path.join(dir_name, nc_files[0])
            ds = xr.open_dataset(nc_path)
            var_map = {'RMSE': 'rmse', 'BIAS': 'bias', 'CORR': 'corr'}
            var_name = var_map.get(col_name)
            if var_name in ds:
                return float(ds[var_name].mean().values)
    except Exception as e:
        pass

    if os.path.exists(path):
        print(f"  [WARN] Failed to extract '{col_name}' for '{season}' from {path}")
    return np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    data_rows = []
    row_labels = []

    for region in REGIONS:
        region_cap = region.capitalize()

        for season in SEASONS:
            row_labels.append(f"{REGION_DISPLAY[region]}_{season}")
            row_data = {}

            for arch in ARCHITECTURES:
                global_name  = f"{arch}_Global"
                regional_name = f"{arch}_{region_cap}"
                display = ARCH_DISPLAY[arch]

                # ---- RMSE ----
                rmse_global_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "rmse", "results",
                    f"seasonal_metrics_{global_name}_pr_{region}_calcul_land_mean_first.csv")
                rmse_reg_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "rmse", "results",
                    f"seasonal_metrics_{regional_name}_pr_{region}_calcul_land_mean_first.csv")

                # ---- BIAS ----
                bias_global_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "bias", "results",
                    f"seasonal_metrics_{global_name}_pr_{region}_calcul_land_mean_first.csv")
                bias_reg_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "bias", "results",
                    f"seasonal_metrics_{regional_name}_pr_{region}_calcul_land_mean_first.csv")

                # ---- CORRELATION ----
                corr_global_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "correlation", "results",
                    f"seasonal_metrics_{global_name}_pr_{region}_calcul_land_per_year.csv")
                corr_reg_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "correlation", "results",
                    f"seasonal_metrics_{regional_name}_pr_{region}_calcul_land_per_year.csv")

                rmse_g  = extract_metric(rmse_global_path,  'RMSE', season)
                rmse_r  = extract_metric(rmse_reg_path,     'RMSE', season)
                # Positive = regional improved (lower RMSE)
                skill_rmse = rmse_g - rmse_r if not np.isnan(rmse_g) else np.nan

                bias_g  = extract_metric(bias_global_path,  'BIAS', season)
                bias_r  = extract_metric(bias_reg_path,     'BIAS', season)
                # Positive = regional improved (lower |bias|)
                skill_bias = abs(bias_g) - abs(bias_r) if not np.isnan(bias_g) else np.nan

                corr_g  = extract_metric(corr_global_path,  'CORR', season)
                corr_r  = extract_metric(corr_reg_path,     'CORR', season)
                # Positive = regional improved (higher correlation)
                skill_corr = corr_r - corr_g if not np.isnan(corr_g) else np.nan

                row_data[f"{display} RMSE"]   = skill_rmse
                row_data[f"{display} r"]      = skill_corr
                row_data[f"{display} |Bias|"] = skill_bias

            data_rows.append(row_data)

    df = pd.DataFrame(data_rows, index=row_labels)

    # Order columns: group by metric, then architecture
    cols = []
    for metric in ["RMSE", "r", "|Bias|"]:
        for arch_d in ["U-Net", "CNN", "ViT"]:
            cols.append(f"{arch_d} {metric}")
    df = df[cols]

    print("\nSkill-change matrix (mean metrics):")
    print(df.to_string())

    # -----------------------------------------------------------------------
    def create_heatmap(df, vmax, suffix):
        plt.figure(figsize=(15, 8))
        sns.set(style='white', font_scale=1.1)

        ax = sns.heatmap(
            df, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,
            annot=df, fmt=".2f", annot_kws={"size": 11},
            linewidths=1.5, linecolor='white'
        )

        # Vertical separators between metric groups
        ax.axvline(x=3, color='black', lw=1.5)
        ax.axvline(x=6, color='black', lw=1.5)

        # Horizontal separators between regions (3 seasons per region)
        for y in [3, 6]:
            ax.axhline(y=y, color='gray', lw=1, linestyle='--')

        # Y-tick labels: season names repeated for each region
        ax.set_yticks(np.arange(0.5, len(row_labels) + 0.5, 1))
        ax.set_yticklabels(['Annual', 'DJF', 'JJA'] * len(REGIONS), rotation=0, fontsize=11)

        # Region labels on the left
        for i, reg in enumerate(REGIONS):
            ax.text(-0.8, i * 3 + 1.5, REGION_DISPLAY[reg],
                    va='center', ha='right', weight='bold', fontsize=13)

        # Metric family headers above the heatmap
        ax.text(1.5, -0.4, 'Error (RMSE)',         ha='center', va='center', fontsize=13, weight='bold')
        ax.text(1.5, -0.15, 'Global - Regional (mm/day)', ha='center', va='center', fontsize=10, style='italic', color='#555555')
        
        ax.text(4.5, -0.4, 'Temporal correlation', ha='center', va='center', fontsize=13, weight='bold')
        ax.text(4.5, -0.15, 'Regional - Global',          ha='center', va='center', fontsize=10, style='italic', color='#555555')
        
        ax.text(7.5, -0.4, 'Absolute bias',        ha='center', va='center', fontsize=13, weight='bold')
        ax.text(7.5, -0.15, '|Global| - |Regional| (mm/day)', ha='center', va='center', fontsize=10, style='italic', color='#555555')

        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Skill Change', size=13, weight='bold')

        # plt.title("Skill Change: Unified-Loss → Regionalized-Loss Training\n(Mean Precipitation Metrics)",
        #           fontsize=15, weight='bold', pad=30)
        plt.ylabel("")
        plt.xlabel("")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        for ext in ['png', 'pdf']:
            out = os.path.join(BASE_RESULTS_DIR, f"regional_added_value_heatmap_{suffix}.{ext}")
            plt.savefig(out, dpi=300, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close()
    # -----------------------------------------------------------------------

    # Main version (capped colorbar for readability)
    create_heatmap(df, vmax=0.25, suffix="main")

    # Supplementary version (full range)
    max_val = np.nanmax(np.abs(df.values))
    create_heatmap(df, vmax=max_val if max_val > 0 else 0.25, suffix="supp")


if __name__ == "__main__":
    # Run from postproc/ directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
