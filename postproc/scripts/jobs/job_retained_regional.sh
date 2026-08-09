#!/bin/bash
#SBATCH --job-name=val_retained_regional
#SBATCH --output=postproc/logs/postproc_retained_regional_%j.log
#SBATCH --error=postproc/logs/postproc_retained_regional_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

source ~/.bashrc
conda activate clean_env_Pytorch

echo "=== Starting Post-Processing for Regional Models ==="
export PYTHONUNBUFFERED=1

cd postproc
CONFIG="configs/config_retained_regional.yaml"

python3 -u src/postproc.py "$CONFIG"
