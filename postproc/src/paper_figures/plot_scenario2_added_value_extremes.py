"""
Author: M. El Aabaribaoune (@um6p)
Description: Computes and plots extreme precipitation indices.
"""


import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    base_dir = "results/scenarios_comparison/2006-01-01_2020-12-31"
    
    regions = ['north', 'north_east', 'east', 'south']
    seasons = ['Annual', 'DJF', 'JJA']
    architectures = ['Unet', 'CNN', 'ViT']
    
    # Store rows data
    r95_data_rows = []
    cdd_data_rows = []
    row_labels = []
    
    for region in regions:
        region_display = region.replace('_', '-').title()
        
        for season in seasons:
            row_label = f"{region_display}_{season}"
            row_labels.append(row_label)
            
            r95_row_data = {}
            cdd_row_data = {}
            
            # Read MSWEP Truth once per region/season
            r95_mswep_path = os.path.join(base_dir, region, "test", "r95_nbEvents_freq", "results", f"seasonal_metrics_mswep_pr_{region}_calcul_land.csv")
            cdd_mswep_path = os.path.join(base_dir, region, "test", "cdd", "cdd_1mm", "results", f"seasonal_metrics_mswep_pr_{region}_calcul_land_per_year.csv")
            
            def extract_metric(path, col_name, season):
                try:
                    df = pd.read_csv(path, index_col=0)
                    return df.loc[season, col_name]
                except Exception as e:
                    if os.path.exists(path):
                        print(f"Failed to extract {col_name} from {path}: {e}")
                    return np.nan
                    
            r95_mswep = extract_metric(r95_mswep_path, 'R95P', season)
            cdd_mswep = extract_metric(cdd_mswep_path, 'CDD', season)
            
            for arch in architectures:
                # Determine file paths
                s1_name = f"{arch}_Scenario1"
                s2_name = f"{arch}_Scenario2"
                
                # R95
                r95_s1_path = os.path.join(base_dir, region, "test", "r95_nbEvents_freq", "results", f"seasonal_metrics_{s1_name}_pr_{region}_calcul_land.csv")
                r95_s2_path = os.path.join(base_dir, region, "test", "r95_nbEvents_freq", "results", f"seasonal_metrics_{s2_name}_pr_{region}_calcul_land.csv")
                
                # CDD
                cdd_s1_path = os.path.join(base_dir, region, "test", "cdd", "cdd_1mm", "results", f"seasonal_metrics_{s1_name}_pr_{region}_calcul_land_per_year.csv")
                cdd_s2_path = os.path.join(base_dir, region, "test", "cdd", "cdd_1mm", "results", f"seasonal_metrics_{s2_name}_pr_{region}_calcul_land_per_year.csv")
                
                r95_s1 = extract_metric(r95_s1_path, 'R95P', season)
                r95_s2 = extract_metric(r95_s2_path, 'R95P', season)
                
                cdd_s1 = extract_metric(cdd_s1_path, 'CDD', season)
                cdd_s2 = extract_metric(cdd_s2_path, 'CDD', season)
                
                # Calculate absolute errors
                error_r95_s1 = abs(r95_s1 - r95_mswep) if not np.isnan(r95_s1) and not np.isnan(r95_mswep) else np.nan
                error_r95_s2 = abs(r95_s2 - r95_mswep) if not np.isnan(r95_s2) and not np.isnan(r95_mswep) else np.nan
                
                error_cdd_s1 = abs(cdd_s1 - cdd_mswep) if not np.isnan(cdd_s1) and not np.isnan(cdd_mswep) else np.nan
                error_cdd_s2 = abs(cdd_s2 - cdd_mswep) if not np.isnan(cdd_s2) and not np.isnan(cdd_mswep) else np.nan
                
                # Skill is positive if Error S2 < Error S1 (reduction in absolute error)
                skill_r95 = error_r95_s1 - error_r95_s2
                skill_cdd = error_cdd_s1 - error_cdd_s2
                
                display_arch = "U-Net" if arch == "Unet" else arch
                
                r95_row_data[f"{display_arch}"] = skill_r95
                cdd_row_data[f"{display_arch}"] = skill_cdd
                
            r95_data_rows.append(r95_row_data)
            cdd_data_rows.append(cdd_row_data)

    df_r95 = pd.DataFrame(r95_data_rows, index=row_labels)
    df_cdd = pd.DataFrame(cdd_data_rows, index=row_labels)
    
    cols = ["U-Net", "CNN", "ViT"]
    df_r95 = df_r95[cols]
    df_cdd = df_cdd[cols]
    
    def format_axes(ax, df, title, vmax):
        sns.heatmap(df, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,
                    annot=df, fmt=".2f", annot_kws={"size": 12}, linewidths=1, linecolor='white', ax=ax, cbar=False)
        
        # Add horizontal lines to separate regions visually
        ax.axhline(y=3, color='gray', lw=1, linestyle='--')
        ax.axhline(y=6, color='gray', lw=1, linestyle='--')
        ax.axhline(y=9, color='gray', lw=1, linestyle='--')
        
        ax.set_yticks(np.arange(0.5, 12.5, 1))
        ax.set_yticklabels(['Annual', 'DJF', 'JJA'] * 4, rotation=0, fontsize=12)
        
        ax.set_title(title, fontsize=16, weight='bold', pad=15)
        ax.set_ylabel("") 
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=13)

    def create_2panel_heatmap(vmax_r95, vmax_cdd, suffix):
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))
        sns.set(style='white', font_scale=1.1)
        
        format_axes(axes[0], df_r95, "a) R95 event frequency skill change", vmax_r95)
        format_axes(axes[1], df_cdd, "b) CDD skill change", vmax_cdd)
        
        # Add Region labels on the left of panel A
        regions_display = ['North', 'North-East', 'East', 'South']
        for i, reg in enumerate(regions_display):
            axes[0].text(-0.8, i * 3 + 1.5, reg, va='center', ha='right', weight='bold', fontsize=14)
            
        fig.subplots_adjust(bottom=0.2, wspace=0.25)
        
        # Add separate colorbars for each panel at the bottom
        cbar_ax1 = fig.add_axes([0.15, 0.06, 0.3, 0.025]) # x, y, width, height
        cbar_ax2 = fig.add_axes([0.57, 0.06, 0.3, 0.025])
        
        sm1 = plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-vmax_r95, vmax=vmax_r95))
        sm2 = plt.cm.ScalarMappable(cmap='RdBu', norm=plt.Normalize(vmin=-vmax_cdd, vmax=vmax_cdd))
        
        cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal')
        cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
        
        cbar_label = 'Relative skill change'
        cb1.set_label(cbar_label, size=12, weight='bold')
        cb2.set_label(cbar_label, size=12, weight='bold')
        
        fig.suptitle("Skill Change in Wet and Dry Extremes", 
                     fontsize=18, weight='bold', y=0.98)
        
        save_path_png = os.path.join(base_dir, f"scenario2_added_value_extremes_heatmap_{suffix}.png")
        save_path_pdf = os.path.join(base_dir, f"scenario2_added_value_extremes_heatmap_{suffix}.pdf")
        
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path_png} and {save_path_pdf}")
        plt.close()

    # Maximum absolute true values
    true_max_r95 = np.nanmax(np.abs(df_r95.values))
    true_max_cdd = np.nanmax(np.abs(df_cdd.values))
    
    # Cap values heuristically for the Main version (avoid ViT extreme degradations stretching the scale)
    limit_r95 = min(true_max_r95, 2.5) if true_max_r95 > 0 else 1.0
    limit_cdd = min(true_max_cdd, 15.0) if true_max_cdd > 0 else 5.0
    
    create_2panel_heatmap(vmax_r95=limit_r95, vmax_cdd=limit_cdd, suffix="main")
    create_2panel_heatmap(vmax_r95=true_max_r95, vmax_cdd=true_max_cdd, suffix="supp")

if __name__ == "__main__":
    main()
