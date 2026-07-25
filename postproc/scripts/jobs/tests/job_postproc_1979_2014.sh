#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=postproc_1979_2014
#SBATCH --output=postproc_1979_2014_%j.log
#SBATCH --error=postproc_1979_2014_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Post-processing Multi-Model Comparison (1979-2014)"
echo "======================================"

python3 -u postproc.py config_1979_2014.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
