#!/bin/bash
# ==============================================================================
# Author: M. El Aabaribaoune (@um6p)
# Description: Job submission script for running pipeline evaluations on the SLURM cluster.
# ==============================================================================

#SBATCH --job-name=eval_postproc
#SBATCH --output=logs/eval_postproc_%j.log
#SBATCH --error=logs/eval_postproc_%j.log
#SBATCH #--partition=gpu
#SBATCH #--gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1
cd ..

echo "======================================"
echo "Evaluating UNet Experiments"
echo "======================================"
python3 -u src/postproc.py configs/experiments/config_unet_experiments.yaml

echo "======================================"
echo "Evaluating ViT Experiments"
echo "======================================"
python3 -u src/postproc.py configs/experiments/config_vit_experiments.yaml

echo "All Postprocessing Completed!"
