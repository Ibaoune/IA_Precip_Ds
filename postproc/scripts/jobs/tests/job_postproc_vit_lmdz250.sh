#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=postproc_vit_lmdz250
#SBATCH --output=postproc_vit_lmdz250_%j.log
#SBATCH --error=postproc_vit_lmdz250_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Post-processing ViT LMDZ 250 Present"
echo "======================================"

python3 -u postproc.py config_vit_lmdz250.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
