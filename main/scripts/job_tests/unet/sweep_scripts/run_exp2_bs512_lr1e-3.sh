#!/bin/bash
#SBATCH --job-name=e2_b512_l1e-3
#SBATCH --output=../logs/sweep_exp2_bs512_lr1e-3_%j.log
#SBATCH --error=../logs/sweep_exp2_bs512_lr1e-3_%j.log
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
echo "Running Sweep: Exp 2, BS 512, LR 1e-3"
echo "======================================"

cd ../../../..
python3 -u train.py configs/unet/tests/sweep/test_exp2_bs512_lr1e-3.yaml
python3 -u eval.py configs/unet/tests/sweep/test_exp2_bs512_lr1e-3.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
