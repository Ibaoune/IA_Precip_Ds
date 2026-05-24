#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=run_inference
#SBATCH --output=logs/out_inference_%j.log
#SBATCH --error=logs/out_inference_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "Running inference with config.yaml..."
python3 -u predict.py config.yaml
