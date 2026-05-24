#!/bin/bash
#SBATCH --job-name=eval_unet_11
#SBATCH --output=logs/eval_unet_11_%j.log
#SBATCH --error=logs/eval_unet_11_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Evaluating UNet Experiment: 11"
echo "======================================"

cd ../../..
python3 -u eval.py configs/unet/tests/test_exp11.yaml

echo "======================================"
