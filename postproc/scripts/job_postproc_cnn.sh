#!/bin/bash
# ==============================================================================
# Author: M. El Aabaribaoune (@um6p)
# Description: Job submission script for running pipeline evaluations on the SLURM cluster.
# ==============================================================================

#SBATCH --job-name=eval_postproc_cnn
#SBATCH --output=logs/eval_postproc_cnn_%j.log
#SBATCH --error=logs/eval_postproc_cnn_%j.log
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch
export PYTHONUNBUFFERED=1
cd /srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc

echo "Evaluating CNN Experiments"
python3 -u src/postproc.py configs/experiments/config_cnn_all.yaml
