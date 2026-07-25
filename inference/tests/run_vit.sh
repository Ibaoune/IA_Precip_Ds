#!/bin/bash
# ==============================================================================
# Author: M. El Aabaribaoune (@um6p)
# Description: Part of the downscaling inference engine.
# ==============================================================================

#SBATCH --job-name=vit_pred
#SBATCH --output=out_vit_%j.log
#SBATCH --error=out_vit_%j.log
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Record start time
start_time=$(date +%s)

echo "Starting ViT Prediction..." 
python3 -u predict.py config_vit.yaml

# Record end time and log total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."
