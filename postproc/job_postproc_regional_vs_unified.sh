#!/bin/bash
#SBATCH --job-name=pp_regional_vs_unified
#SBATCH --output=logs/postproc_regional_vs_unified_%j.log
#SBATCH --error=logs/postproc_regional_vs_unified_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Post-processing Regional vs Unified Comparison"
echo "======================================"

echo "Starting regional comparison post-processing suite."
python3 -u postproc.py tests/config_regional_vs_unified.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
