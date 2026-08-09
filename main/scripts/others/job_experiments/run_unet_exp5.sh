#!/bin/bash
#SBATCH --job-name=unet_exp5
#SBATCH --output=logs/unet_exp5_%j.log
#SBATCH --error=logs/unet_exp5_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running Experiment: unet_exp5"
echo "======================================"

cd ../..
python3 -u train.py configs/experiments/unet_exp5.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
