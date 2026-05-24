#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=e1_b512_l1e-4
#SBATCH --output=../logs/sweep_exp1_bs512_lr1e-4_%j.log
#SBATCH --error=../logs/sweep_exp1_bs512_lr1e-4_%j.log
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
echo "Running Sweep: Exp 1, BS 512, LR 1e-4"
echo "======================================"

cd ../../../..
python3 -u train.py configs/unet/tests/sweep/test_exp1_bs512_lr1e-4.yaml
python3 -u eval.py configs/unet/tests/sweep/test_exp1_bs512_lr1e-4.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
