#!/bin/bash
#SBATCH --job-name=vit_exp8
#SBATCH --output=logs/vit_exp8_%j.log
#SBATCH --error=logs/vit_exp8_%j.log
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
echo "Running Experiment: vit_exp8"
echo "======================================"

cd ../..
python3 -u train.py configs/experiments/vit_exp8.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
