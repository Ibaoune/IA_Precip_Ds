#!/bin/bash
#SBATCH --job-name=eval_unet_exp2
#SBATCH --output=logs/eval_unet_exp2_%j.log
#SBATCH --error=logs/eval_unet_exp2_%j.log
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
echo "Running Evaluation for: unet_exp2"
echo "======================================"

cd ../..
python3 -u eval.py configs/experiments/unet_exp2.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
