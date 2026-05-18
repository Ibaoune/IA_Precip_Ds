#!/bin/bash

# ==========================================================
# Script: explore.sh
# Description:
#     Executes the dataset exploration pipeline to generate
#     preliminary climatology and intensity plots.
# ==========================================================

# Set project root in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "----------------------------------------------------"
echo "   Starting Dataset Exploration Pipeline"
echo "   (Working Root: $(pwd))"
echo "----------------------------------------------------"

# Activate conda environment
source /home/hassan/anaconda3/etc/profile.d/conda.sh
conda activate env_torch

# Run the exploration script
echo "[STEP 1/1] Generating Exploration Plots..."
python3 src/exploration.py --config config_explore.yaml

echo ""
echo "----------------------------------------------------"
echo "   Exploration Completed!"
echo "   Results Saved in: results/"
echo "----------------------------------------------------"
