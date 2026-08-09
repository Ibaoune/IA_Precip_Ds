#!/bin/bash
# Script to automate the generation of heatmaps after postproc finishes
set -e

source ~/.bashrc
conda activate clean_env_Pytorch

cd /srv/data/mohammad.elaabaribao/work/papers/downscaling

echo "Wait for SLURM job to finish... (Check manually with squeue -j 7415136)"
# We won't block here in bash, we'll let the Python agent wait.

cd postproc
export PYTHONPATH=src

echo "[INFO] Running regional summary script..."
python3 -u src/summaries/regional_summary.py configs/config_retained_regional.yaml --all-periods

echo "[INFO] Running heatmap script for mean metrics..."
python3 src/paper_figures/plot_regional_added_value.py

echo "[INFO] Running heatmap script for extremes..."
python3 src/paper_figures/plot_regional_added_value_extremes.py

echo "Done!"
