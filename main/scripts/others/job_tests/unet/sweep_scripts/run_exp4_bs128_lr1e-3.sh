#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=e4_b128_l1e-3
#SBATCH --output=../logs/sweep_exp4_bs128_lr1e-3_%j.log
#SBATCH --error=../logs/sweep_exp4_bs128_lr1e-3_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: \$SLURM_JOB_ID"
echo "Running Sweep: Exp 4, BS 128, LR 1e-3"
echo "======================================"

cd ../../../..
python3 -u train.py configs/unet/tests/sweep/test_exp4_bs128_lr1e-3.yaml
python3 -u eval.py configs/unet/tests/sweep/test_exp4_bs128_lr1e-3.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
