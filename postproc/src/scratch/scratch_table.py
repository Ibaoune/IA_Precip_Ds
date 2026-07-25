"""
Author: M. El Aabaribaoune (@um6p)
Description: Part of the post-processing and evaluation pipeline for the downscaling project.
"""

import os
import pandas as pd

base_dir = "/srv/data/mohammad.elaabaribao/work/interns/y2026/code/era5Tomswep/precip/postproc/results/cnn_unet_glm_vit/test"

models = ["unet", "cnn", "glm", "vit"]
periods = ["Annual", "DJF", "JJA"]

# Map to store data: data[model][period][metric] = value
data = {m: {p: {} for p in periods} for m in models}

metrics_info = {
    "Bias": ("bias", "mean_first", "BIAS"),
    "RMSE": ("rmse", "mean_first", "RMSE"),
    "Correlation": ("correlation", "per_year", "CORRELATION") # Column name is usually the metric name uppercased, let's read the first column name
}

for metric_name, (folder, strategy, col_fallback) in metrics_info.items():
    for model in models:
        csv_path = os.path.join(base_dir, folder, "results", f"seasonal_metrics_{model}_pr_allmorr_calcul_land_{strategy}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            col = df.columns[0]
            for period in periods:
                if period in df.index:
                    val = df.loc[period, col]
                    data[model][period][metric_name] = f"{val:.3f}"
                else:
                    data[model][period][metric_name] = "N/A"
        else:
            for period in periods:
                data[model][period][metric_name] = "File missing"

output_md = os.path.join(base_dir, "mean_metrics_summary.md")
output_csv = os.path.join(base_dir, "mean_metrics_summary.csv")

with open(output_md, "w") as f:
    f.write("# Mean Metrics Comparison (Bias, RMSE, Correlation)\n\n")
    f.write("### Organized by Model and Period\n\n")
    f.write("| Model | Period | Bias (mm/day) | RMSE (mm/day) | Correlation |\n")
    f.write("|-------|--------|---------------|---------------|-------------|\n")
    for model in models:
        for i, period in enumerate(periods):
            m_disp = model.upper() if i == 0 else ""
            bias = data[model][period].get("Bias", "-")
            rmse = data[model][period].get("RMSE", "-")
            corr = data[model][period].get("Correlation", "-")
            f.write(f"| {m_disp} | {period} | {bias} | {rmse} | {corr} |\n")

    f.write("\n\n### Organized by Period and Metric\n\n")
    f.write("| Period | Metric | UNET | CNN | GLM | ViT |\n")
    f.write("|--------|--------|------|-----|-----|-----|\n")
    for period in periods:
        for i, metric in enumerate(["Bias", "RMSE", "Correlation"]):
            p_disp = period if i == 0 else ""
            row = f"| {p_disp} | {metric} |"
            for model in models:
                val = data[model][period].get(metric, "-")
                row += f" {val} |"
            f.write(row + "\n")

# Also save as CSV for Excel
with open(output_csv, "w") as f:
    f.write("Model,Period,Bias,RMSE,Correlation\n")
    for model in models:
        for period in periods:
            bias = data[model][period].get("Bias", "-")
            rmse = data[model][period].get("RMSE", "-")
            corr = data[model][period].get("Correlation", "-")
            f.write(f"{model.upper()},{period},{bias},{rmse},{corr}\n")

print(f"Tables saved to {output_md} and {output_csv}")
