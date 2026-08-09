#!/bin/bash
# ==============================================================================
# Author: M. El Aabaribaoune (@um6p)
# Description: Job submission script for generating LMDZ paper figures.
# ==============================================================================

#SBATCH --job-name=val_lmdz_figures_retained
#SBATCH --output=postproc/logs/postproc_lmdz_figures_retained_%j.log
#SBATCH --error=postproc/logs/postproc_lmdz_figures_retained_%j.log
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "=== Generating LMDZ Paper Figures ==="
export PYTHONUNBUFFERED=1

cd postproc
export PYTHONPATH=".:$PYTHONPATH"

echo "Running plot_lmdz_paper_figures.py..."
python3 -u src/paper_figures/plot_lmdz_paper_figures.py

echo "Running plot_lmdz_annual_figures.py..."
python3 -u src/paper_figures/plot_lmdz_annual_figures.py

echo "=== LMDZ Paper Figures Generation Complete ==="
