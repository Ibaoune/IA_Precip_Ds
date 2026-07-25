"""
Author: M. El Aabaribaoune (@um6p)
Description: Generates custom, publication-ready figures for the research paper.
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
    data_rows = []
    row_labels = []
    
    for region in regions:
        region_display = region.replace('_', '-').title()
        
        for season in seasons:
            row_label = f"{region_display}_{season}"
            row_labels.append(row_label)
            
            row_data = {}
            for arch in architectures:
                # Determine file paths
                s1_name = f"{arch}_Scenario1"
                s2_name = f"{arch}_Scenario2"
                
                # RMSE (from rmse folder)
                rmse_s1_path = os.path.join(base_dir, region, "test", "rmse", "results", f"seasonal_metrics_{s1_name}_pr_{region}_calcul_land_mean_first.csv")
                rmse_s2_path = os.path.join(base_dir, region, "test", "rmse", "results", f"seasonal_metrics_{s2_name}_pr_{region}_calcul_land_mean_first.csv")
                
                # BIAS (from bias folder)
                bias_s1_path = os.path.join(base_dir, region, "test", "bias", "results", f"seasonal_metrics_{s1_name}_pr_{region}_calcul_land_mean_first.csv")
                bias_s2_path = os.path.join(base_dir, region, "test", "bias", "results", f"seasonal_metrics_{s2_name}_pr_{region}_calcul_land_mean_first.csv")
                
                # CORR (from correlation folder)
                corr_s1_path = os.path.join(base_dir, region, "test", "correlation", "results", f"seasonal_metrics_{s1_name}_pr_{region}_calcul_land_per_year.csv")
                corr_s2_path = os.path.join(base_dir, region, "test", "correlation", "results", f"seasonal_metrics_{s2_name}_pr_{region}_calcul_land_per_year.csv")
                
                # Function to extract metric
                def extract_metric(path, col_name, season):
                    try:
                        df = pd.read_csv(path, index_col=0)
                        return df.loc[season, col_name]
                    except Exception as e:
                        if os.path.exists(path):
                            print(f"Failed to extract {col_name} from {path}: {e}")
                        return np.nan
                
                rmse_s1 = extract_metric(rmse_s1_path, 'RMSE', season)
                rmse_s2 = extract_metric(rmse_s2_path, 'RMSE', season)
                # Positive means Scenario 2 improved (lower RMSE)
                skill_rmse = rmse_s1 - rmse_s2 if not np.isnan(rmse_s1) else np.nan
                
                bias_s1 = extract_metric(bias_s1_path, 'BIAS', season)
                bias_s2 = extract_metric(bias_s2_path, 'BIAS', season)
                # Positive means Scenario 2 improved (lower absolute Bias)
                skill_bias = abs(bias_s1) - abs(bias_s2) if not np.isnan(bias_s1) else np.nan
                
                corr_s1 = extract_metric(corr_s1_path, 'CORR', season)
                corr_s2 = extract_metric(corr_s2_path, 'CORR', season)
                # Positive means Scenario 2 improved (higher correlation)
                skill_corr = corr_s2 - corr_s1 if not np.isnan(corr_s1) else np.nan
                
                display_arch = "U-Net" if arch == "Unet" else arch
                
                row_data[f"{display_arch} RMSE"] = skill_rmse
                row_data[f"{display_arch} r"] = skill_corr
                row_data[f"{display_arch} |Bias|"] = skill_bias
                
            data_rows.append(row_data)

    df = pd.DataFrame(data_rows, index=row_labels)
    
    # Reorder columns to group by metric
    cols = []
    for metric in ["RMSE", "r", "|Bias|"]:
        for arch in ["U-Net", "CNN", "ViT"]:
            cols.append(f"{arch} {metric}")
            
    df = df[cols]
    
    # Generate two versions: Main (limited colorbar) and Supp (full colorbar)
    def create_heatmap(df, vmax, suffix):
        plt.figure(figsize=(15, 10))
        sns.set(style='white', font_scale=1.1)
        
        ax = sns.heatmap(df, cmap='RdBu', center=0, vmin=-vmax, vmax=vmax,
                         annot=df, fmt=".2f", annot_kws={"size": 11}, linewidths=1.5, linecolor='white')
        
        # Thinner vertical lines
        ax.axvline(x=3, color='black', lw=1)
        ax.axvline(x=6, color='black', lw=1)
        
        # Add horizontal lines to separate regions visually
        ax.axhline(y=3, color='gray', lw=1, linestyle='--')
        ax.axhline(y=6, color='gray', lw=1, linestyle='--')
        ax.axhline(y=9, color='gray', lw=1, linestyle='--')
        
        # Custom Y-axis hierarchical labeling
        ax.set_yticks(np.arange(0.5, 12.5, 1))
        ax.set_yticklabels(['Annual', 'DJF', 'JJA'] * 4, rotation=0, fontsize=11)
        
        # Add Region labels as group headers on the left
        regions_display = ['North', 'North-East', 'East', 'South']
        for i, reg in enumerate(regions_display):
            # Place region name in the middle of each 3-row block, to the left of the y-axis
            ax.text(-0.8, i * 3 + 1.5, reg, va='center', ha='right', weight='bold', fontsize=13)
        
        # Add Top Headers for Metric Families
        ax.text(1.5, -0.5, 'Error metrics', ha='center', va='center', fontsize=14, weight='bold')
        ax.text(4.5, -0.5, 'Temporal correlation', ha='center', va='center', fontsize=14, weight='bold')
        ax.text(7.5, -0.5, 'Absolute bias', ha='center', va='center', fontsize=14, weight='bold')
        
        # Customize the colorbar
        colorbar = ax.collections[0].colorbar
        colorbar.set_label('Relative skill change (Positive = improvement; Negative = degradation)', size=12, weight='bold')
        
        # Main Title
        plt.title("Skill Change from Unified to Regionalized-Loss Training", 
                  fontsize=16, weight='bold', pad=30)
        plt.ylabel("") # Remove default ylabel since we have custom group labels
        plt.xlabel("") # Remove default xlabel
        
        # Make column labels horizontal or slightly angled for better reading
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        save_path_png = os.path.join(base_dir, f"scenario2_added_value_heatmap_{suffix}.png")
        save_path_pdf = os.path.join(base_dir, f"scenario2_added_value_heatmap_{suffix}.pdf")
        
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path_png} and {save_path_pdf}")
        plt.close()

    # Create Main Paper Version (limited saturation for better visibility of small changes)
    create_heatmap(df, vmax=0.25, suffix="main")
    
    # Create Supplementary Version (full range, captures extreme ViT degradations)
    max_val = np.nanmax(np.abs(df.values))
    create_heatmap(df, vmax=max_val, suffix="supp")

if __name__ == "__main__":
    main()
