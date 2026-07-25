"""
Author: M. El Aabaribaoune (@um6p)
Description: Part of the post-processing and evaluation pipeline for the downscaling project.
"""

import os
import pandas as pd
import numpy as np

base_dir = "/srv/data/mohammad.elaabaribao/work/interns/y2026/code/era5Tomswep/precip/postproc/results/cnn_unet_glm_vit/test"

models = ["unet", "cnn", "glm", "vit"]
display_models = {"unet": "U-Net", "cnn": "CNN", "glm": "GLM", "vit": "ViT"}
periods = ["Annual", "DJF", "JJA"]
metrics = ["Bias", "RMSE", "Correlation"]

# Map to store data: data[model][metric][period] = float_value
data = {m: {met: {p: np.nan for p in periods} for met in metrics} for m in models}

metrics_info = {
    "Bias": ("bias", "mean_first", "BIAS"),
    "RMSE": ("rmse", "mean_first", "RMSE"),
    "Correlation": ("correlation", "per_year", "CORRELATION")
}

# Load data
for metric_name, (folder, strategy, col_fallback) in metrics_info.items():
    for model in models:
        csv_path = os.path.join(base_dir, folder, "results", f"seasonal_metrics_{model}_pr_allmorr_calcul_land_{strategy}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            col = df.columns[0]
            for period in periods:
                if period in df.index:
                    val = df.loc[period, col]
                    data[model][metric_name][period] = float(val)

# Find best values for bolding
best_vals = {met: {p: None for p in periods} for met in metrics}
for metric in metrics:
    for period in periods:
        vals = [data[m][metric][period] for m in models if not np.isnan(data[m][metric][period])]
        if vals:
            if metric == "Bias":
                # closest to 0
                best_vals[metric][period] = min(vals, key=abs)
            elif metric == "RMSE":
                # lowest
                best_vals[metric][period] = min(vals)
            elif metric == "Correlation":
                # highest
                best_vals[metric][period] = max(vals)

# Formatting function
def format_val(val, metric, period):
    if np.isnan(val):
        return "-"
    
    # Format number
    v_str = f"{val:.2f}"
    
    # Check if best
    is_best = False
    best_v = best_vals[metric][period]
    if best_v is not None and abs(val - best_v) < 1e-5:
        is_best = True
        
    if is_best:
        return f"<b>{v_str}</b>" # HTML bold for markdown
    return v_str

def format_val_csv(val):
    if np.isnan(val):
        return "-"
    return f"{val:.2f}"

output_md = os.path.join(base_dir, "mean_metrics_summary.md")
output_csv = os.path.join(base_dir, "mean_metrics_summary.csv")

# Generate HTML Table for Markdown
html_table = []
html_table.append('<table border="1" style="text-align:center; border-collapse: collapse;">')
# Header Row 1
html_table.append('  <tr>')
html_table.append('    <th rowspan="2">Model</th>')
for metric in metrics:
    html_table.append(f'    <th colspan="{len(periods)}">{metric}</th>')
html_table.append('  </tr>')
# Header Row 2
html_table.append('  <tr>')
for metric in metrics:
    for period in periods:
        html_table.append(f'    <th><i>{period}</i></th>')
html_table.append('  </tr>')

# Data Rows
for model in models:
    html_table.append('  <tr>')
    html_table.append(f'    <td>{display_models[model]}</td>')
    for metric in metrics:
        for period in periods:
            val = data[model][metric][period]
            html_table.append(f'    <td>{format_val(val, metric, period)}</td>')
    html_table.append('  </tr>')

html_table.append('</table>')

with open(output_md, "w") as f:
    f.write("# Mean Metrics Comparison\n\n")
    f.write("\n".join(html_table))
    f.write("\n")

# Generate CSV format
with open(output_csv, "w") as f:
    # Row 1
    f.write("Model")
    for metric in metrics:
        f.write(f",{metric}")
        f.write("," * (len(periods)-1))
    f.write("\n")
    
    # Row 2
    f.write("")
    for metric in metrics:
        for period in periods:
            f.write(f",{period}")
    f.write("\n")
    
    # Data
    for model in models:
        f.write(f"{display_models[model]}")
        for metric in metrics:
            for period in periods:
                val = data[model][metric][period]
                f.write(f",{format_val_csv(val)}")
        f.write("\n")

print(f"Formatted tables saved to {output_md} and {output_csv}")
