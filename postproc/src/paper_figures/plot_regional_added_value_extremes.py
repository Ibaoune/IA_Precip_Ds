"""
Author: M. El Aabaribaoune (@um6p)
Description: Two-panel heatmap of skill change (Global → Regionalized-Loss) for precipitation
             extremes (R95 event frequency and CDD) across sub-regions and seasons.

Adapted from plot_scenario2_added_value_extremes.py for the Retained_Regional_Evaluation experiment.

Mapping:
    Scenario1 (global)    →  {Arch}_Global
    Scenario2 (regional)  →  {Arch}_{Region}   (e.g. CNN_North, Unet_South, …)

Inputs (seasonal CSV files per sub-region produced by postproc.py + regional_summary.py):
    results/Retained_Regional_Evaluation/<dates>/<region>/test/r95_nbEvents_freq/results/
        seasonal_metrics_{model}_pr_{region}_calcul_land.csv          (col: R95P)
        seasonal_metrics_mswep_pr_{region}_calcul_land.csv            (reference)
    results/Retained_Regional_Evaluation/<dates>/<region>/test/cdd/cdd_1mm/results/
        seasonal_metrics_{model}_pr_{region}_calcul_land_per_year.csv (col: CDD)
        seasonal_metrics_mswep_pr_{region}_calcul_land_per_year.csv   (reference)

Outputs:
    results/Retained_Regional_Evaluation/<dates>/regional_added_value_extremes_heatmap_main.pdf/.png
    results/Retained_Regional_Evaluation/<dates>/regional_added_value_extremes_heatmap_supp.pdf/.png
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

REGIONS      = ['north', 'north_east', 'east', 'south']
SEASONS      = ['Annual', 'DJF', 'JJA']
ARCHITECTURES = ['Unet', 'CNN', 'Vit']
ARCH_DISPLAY  = {'Unet': 'U-Net', 'CNN': 'CNN', 'Vit': 'ViT'}
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

    try:
        dir_name = os.path.dirname(path)
        base_name = os.path.basename(path).replace("seasonal_metrics_", "").replace(".csv", "")
        parts = base_name.split("_pr_")
        model_part = parts[0]
        
        nc_files = [f for f in os.listdir(dir_name) if f.startswith(model_part + "_pr_") and f.endswith(f"_{season}.nc")]
        if nc_files:
            nc_path = os.path.join(dir_name, nc_files[0])
            ds = xr.open_dataset(nc_path)
            var_map = {'R95P': 'r95p', 'CDD': 'cdd'}
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
    r95_data_rows = []
    cdd_data_rows = []
    row_labels    = []

    for region in REGIONS:
        region_cap = region.capitalize()

        for season in SEASONS:
            row_labels.append(f"{REGION_DISPLAY[region]}_{season}")
            r95_row = {}
            cdd_row = {}

            # ---- Reference (MSWEP) ----
            r95_mswep_path = os.path.join(
                BASE_RESULTS_DIR, region, "test", "r95_nbEvents_freq", "results",
                f"seasonal_metrics_mswep_pr_{region}_calcul_land.csv")
            cdd_mswep_path = os.path.join(
                BASE_RESULTS_DIR, region, "test", "cdd", "cdd_1mm", "results",
                f"seasonal_metrics_mswep_pr_{region}_calcul_land_per_year.csv")

            r95_mswep = extract_metric(r95_mswep_path, 'R95P', season)
            cdd_mswep = extract_metric(cdd_mswep_path, 'CDD',  season)

            for arch in ARCHITECTURES:
                global_name   = f"{arch}_Global"
                regional_name = f"{arch}_{region_cap}"
                display = ARCH_DISPLAY[arch]

                # ---- R95 ----
                r95_global_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "r95_nbEvents_freq", "results",
                    f"seasonal_metrics_{global_name}_pr_{region}_calcul_land.csv")
                r95_reg_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "r95_nbEvents_freq", "results",
                    f"seasonal_metrics_{regional_name}_pr_{region}_calcul_land.csv")

                # ---- CDD ----
                cdd_global_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "cdd", "cdd_1mm", "results",
                    f"seasonal_metrics_{global_name}_pr_{region}_calcul_land_per_year.csv")
                cdd_reg_path = os.path.join(
                    BASE_RESULTS_DIR, region, "test", "cdd", "cdd_1mm", "results",
                    f"seasonal_metrics_{regional_name}_pr_{region}_calcul_land_per_year.csv")

                r95_g = extract_metric(r95_global_path, 'R95P', season)
                r95_r = extract_metric(r95_reg_path,    'R95P', season)
                cdd_g = extract_metric(cdd_global_path, 'CDD',  season)
                cdd_r = extract_metric(cdd_reg_path,    'CDD',  season)

                # Absolute error vs MSWEP
                err_r95_g = abs(r95_g - r95_mswep) if not (np.isnan(r95_g) or np.isnan(r95_mswep)) else np.nan
                err_r95_r = abs(r95_r - r95_mswep) if not (np.isnan(r95_r) or np.isnan(r95_mswep)) else np.nan
                err_cdd_g = abs(cdd_g - cdd_mswep) if not (np.isnan(cdd_g) or np.isnan(cdd_mswep)) else np.nan
                err_cdd_r = abs(cdd_r - cdd_mswep) if not (np.isnan(cdd_r) or np.isnan(cdd_mswep)) else np.nan

                # Positive = regional improved (lower absolute error)
                r95_row[display] = err_r95_g - err_r95_r
                cdd_row[display] = err_cdd_g - err_cdd_r

            r95_data_rows.append(r95_row)
            cdd_data_rows.append(cdd_row)

    df_r95 = pd.DataFrame(r95_data_rows, index=row_labels)[["U-Net", "CNN", "ViT"]]
    df_cdd = pd.DataFrame(cdd_data_rows, index=row_labels)[["U-Net", "CNN", "ViT"]]

    print("\nSkill-change matrix (R95 event frequency):")
    print(df_r95.to_string())
    print("\nSkill-change matrix (CDD):")
    print(df_cdd.to_string())

    # -----------------------------------------------------------------------
    def format_axes(ax, df, title, subtitle, vmax):
        sns.heatmap(
            df, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,
            annot=df, fmt=".2f", annot_kws={"size": 12},
            linewidths=1, linecolor='white', ax=ax, cbar=False
        )
        for y in [3, 6]:
            ax.axhline(y=y, color='gray', lw=1, linestyle='--')

        ax.set_yticks(np.arange(0.5, len(row_labels) + 0.5, 1))
        ax.set_yticklabels(['Annual', 'DJF', 'JJA'] * len(REGIONS), rotation=0, fontsize=12)
        ax.set_title(title, fontsize=15, weight='bold', pad=40)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center', va='bottom', fontsize=10, style='italic', color='#555555')
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=13)

    def create_2panel_heatmap(vmax_r95, vmax_cdd, suffix):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        sns.set(style='white', font_scale=1.1)

        format_axes(axes[0], df_r95, "a) R95 event frequency – skill change", "|Global Error| - |Regional Error| (events/year)", vmax_r95)
        format_axes(axes[1], df_cdd, "b) CDD – skill change", "|Global Error| - |Regional Error| (days)", vmax_cdd)

        # Region labels on the left of panel A
        for i, reg in enumerate(REGIONS):
            axes[0].text(-0.8, i * 3 + 1.5, REGION_DISPLAY[reg],
                         va='center', ha='right', weight='bold', fontsize=14)

        fig.subplots_adjust(bottom=0.22, wspace=0.25)

        cbar_ax1 = fig.add_axes([0.15, 0.07, 0.30, 0.025])
        cbar_ax2 = fig.add_axes([0.57, 0.07, 0.30, 0.025])

        sm1 = plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(-vmax_r95, vmax_r95))
        sm2 = plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(-vmax_cdd, vmax_cdd))
        cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal')
        cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
        for cb in [cb1, cb2]:
            cb.set_label('Skill Change', size=13, weight='bold')

        # fig.suptitle("Skill Change in Wet and Dry Extremes\n(Unified-Loss → Regionalized-Loss Training)",
        #              fontsize=16, weight='bold', y=0.98)

        for ext in ['png', 'pdf']:
            out = os.path.join(BASE_RESULTS_DIR, f"regional_added_value_extremes_heatmap_{suffix}.{ext}")
            plt.savefig(out, dpi=300, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close()
    # -----------------------------------------------------------------------

    true_max_r95 = np.nanmax(np.abs(df_r95.values)) if np.any(~np.isnan(df_r95.values)) else 1.0
    true_max_cdd = np.nanmax(np.abs(df_cdd.values)) if np.any(~np.isnan(df_cdd.values)) else 5.0

    limit_r95 = min(true_max_r95, 2.5)
    limit_cdd = min(true_max_cdd, 15.0)

    create_2panel_heatmap(vmax_r95=limit_r95, vmax_cdd=limit_cdd, suffix="main")
    create_2panel_heatmap(vmax_r95=true_max_r95, vmax_cdd=true_max_cdd, suffix="supp")


if __name__ == "__main__":
    # Run from postproc/ directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    main()
