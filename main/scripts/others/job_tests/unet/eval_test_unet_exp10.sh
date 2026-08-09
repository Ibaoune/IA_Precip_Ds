#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=eval_unet_10
#SBATCH --output=logs/eval_unet_10_%j.log
#SBATCH --error=logs/eval_unet_10_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Evaluating UNet Experiment: 10"
echo "======================================"

cd ../../..
python3 -u eval.py configs/unet/tests/test_exp10.yaml

echo "======================================"
