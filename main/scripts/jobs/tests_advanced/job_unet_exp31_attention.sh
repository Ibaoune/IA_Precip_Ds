#!/bin/bash
#SBATCH --job-name=unet_exp31_attention
#SBATCH --output=unet_exp31_attention_%j.log
#SBATCH --error=unet_exp31_attention_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch
export PYTHONUNBUFFERED=1

CONFIG="../../../configs/unet/tests/tests_advanced_arch/config_unet_exp31_attention.yaml"

echo "Starting evaluation for $CONFIG"
python3 -u ../../../eval.py "$CONFIG"
