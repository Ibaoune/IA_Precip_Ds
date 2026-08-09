#!/bin/bash
#SBATCH --job-name=unet_1
#SBATCH --output=logs/unet_1_%j.log
#SBATCH --error=logs/unet_1_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running UNet Experiment: 1"
echo "======================================"

cd ../..
python3 -u train.py configs/unet/tests/test_exp1.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
