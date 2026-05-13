#!/bin/bash

# ==========================================================
# Script: evaluate.sh
# Description:
#     Executes climate metrics step-by-step using the new
#     hierarchical structure and configuration system.
# ==========================================================

# Set project root in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "----------------------------------------------------"
echo "   Starting Post-Processing Evaluation Pipeline"
echo "   (Working Root: $(pwd))"
echo "----------------------------------------------------"
# acitvate conda
source /home/hassan/anaconda3/etc/profile.d/conda.sh
conda activate env_torch

# Execute the Master Post-Processing Runner
python3 postproc.py

echo ""
echo "----------------------------------------------------"
echo "   Post-Processing Completed!"
echo "   Results Saved in: results/"
echo "----------------------------------------------------"
