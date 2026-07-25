#!/bin/bash
# ==============================================================================
# Author: M. El Aabaribaoune (@um6p)
# Description: Part of the downscaling inference engine.
# ==============================================================================

#SBATCH --job-name=cnn_lmdz250_fix
#SBATCH --output=logs/out_cnn_fix_%j.log
#SBATCH --error=logs/out_cnn_fix_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "Re-running CNN Exp5 for LMDZ 250..."
# Remove corrupted file if it exists
rm -f results/1979_2014/cnn_exp5/cnn_lmdz_250_present_true.nc
python3 -u predict.py config_cnn_exp5_1979_2014.yaml > logs/cnn_exp5_lmdz250_retry.log 2>&1

echo "Done!"
