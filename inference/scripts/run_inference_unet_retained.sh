#!/bin/bash
# ==============================================================================
# Author: M. El Aabaribaoune (@um6p)
# Description: Job submission script for running inference on the SLURM cluster
#              for the retained U-Net model.
# ==============================================================================

#SBATCH --job-name=run_inference_unet_retained
#SBATCH --output=logs/out_inference_unet_retained_%j.log
#SBATCH --error=logs/out_inference_unet_retained_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch

cd inference
echo "Running inference with config_unet_exp32_parallel_1979_2014.yaml..."
python3 -u src/predict.py tests/config_unet_exp32_parallel_1979_2014.yaml
