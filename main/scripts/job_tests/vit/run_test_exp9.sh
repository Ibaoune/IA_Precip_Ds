#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=test_exp9
#SBATCH --output=test_exp9_%j.log
#SBATCH --error=test_exp9_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

CONFIG_PATH="../../../configs/vit/tests/test_exp9.yaml"

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running test for config: $CONFIG_PATH"
echo "======================================"

echo "[INFO] Running training..."
python3 -u ../../../train.py "$CONFIG_PATH"

echo "[INFO] Running validation..."
python3 -u ../../../eval.py "$CONFIG_PATH"

echo "======================================"
echo "Job completed."
echo "======================================"
