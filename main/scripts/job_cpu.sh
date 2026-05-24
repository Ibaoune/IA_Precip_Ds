#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)

#SBATCH --job-name=vit_T01
#SBATCH --output=vit_T01%j.log
#SBATCH --error=vit_T01%j.log
#SBATCH --nodes=1
#SBATCH --time=36:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

########################################
# USER SHOULD SET THESE TO yes OR no
########################################
train="yes"
validation="yes"
########################################

# Activate Conda environment
#source /home/hassan/anaconda3/etc/profile.d/conda.sh
conda activate clean_env_Pytorch

# Ensure Python logs are flushed immediately
export PYTHONUNBUFFERED=1

# Record start time
start_time=$(date +%s)

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Train: $train | Validation: $validation"
echo "======================================"

if [[ "$train" == "yes" ]]; then
    echo "[INFO] Running training..."
    python3 -u ../train.py ../configs/vit/config.yaml
fi

if [[ "$validation" == "yes" ]]; then
    echo "[INFO] Running validation..."
    python3 -u ../eval.py ../configs/vit/config.yaml
fi

if [[ "$train" != "yes" && "$validation" != "yes" ]]; then
    echo "[WARNING] Neither training nor validation selected."
    echo "Set train=\"yes\" and/or validation=\"yes\"."
fi

# Record end time
end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"