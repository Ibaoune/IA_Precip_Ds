#!/bin/bash
#SBATCH --job-name=eval_vit_retained
#SBATCH --output=logs/eval_vit_retained_%j.log
#SBATCH --error=logs/eval_vit_retained_%j.log
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
echo "Running Evaluation for: vit_retained"
echo "======================================"

cd ../..
python3 -u eval.py configs/vit/retained/config.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
