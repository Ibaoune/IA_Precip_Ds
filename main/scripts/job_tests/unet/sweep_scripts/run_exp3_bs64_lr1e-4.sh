#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=e3_b64_l1e-4
#SBATCH --output=../logs/sweep_exp3_bs64_lr1e-4_%j.log
#SBATCH --error=../logs/sweep_exp3_bs64_lr1e-4_%j.log
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
echo "Running Sweep: Exp 3, BS 64, LR 1e-4"
echo "======================================"

cd ../../../..
python3 -u train.py configs/unet/tests/sweep/test_exp3_bs64_lr1e-4.yaml
python3 -u eval.py configs/unet/tests/sweep/test_exp3_bs64_lr1e-4.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
