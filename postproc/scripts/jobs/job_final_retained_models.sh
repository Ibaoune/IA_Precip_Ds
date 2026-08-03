#!/bin/bash
#SBATCH --job-name=val_final_retained
#SBATCH --output=postproc/logs/postproc_final_retained_%j.log
#SBATCH --error=postproc/logs/postproc_final_retained_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

source ~/.bashrc
conda activate clean_env_Pytorch

echo "=== Starting Final Post-Processing for All Retained Models (GLM, CNN, Unet, Vit) ==="
export PYTHONUNBUFFERED=1

cd postproc
CONFIG="configs/config_final_retained_models.yaml"

python3 -u src/postproc.py "$CONFIG"
