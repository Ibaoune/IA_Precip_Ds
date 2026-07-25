#!/bin/bash
# ==============================================================================
# Author: M. El Aabaribaoune (@um6p)
# Description: Part of the downscaling inference engine.
# ==============================================================================

#SBATCH --job-name=config_cnn_exp3
#SBATCH --output=out_config_cnn_exp3_%j.log
#SBATCH --error=out_config_cnn_exp3_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "Running config_cnn_exp3.yaml..."
python3 -u predict.py config_cnn_exp3.yaml
