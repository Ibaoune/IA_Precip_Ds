#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)

#SBATCH --job-name=vit_T01_gpu
#SBATCH --output=vit_T01_gpu%j.log
#SBATCH --error=vit_T01_gpu%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

######################################## # USER SHOULD SET THESE TO yes OR no
######################################## train="yes"
validation="yes"
######################################## # Activate Conda environment
source ~/.bashrc
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

######################################## # GPU INFO
######################################## echo "[INFO] Allocated GPU(s):"
nvidia-smi

######################################## # GPU monitoring (background)
######################################## gpu_log="gpu_usage_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
 --format=csv,nounits,noheader \
 --loop=60 > "$gpu_log" &

######################################## # CONFIG PATH (same style as CPU job)
######################################## CONFIG="../configs/vit/config.yaml"

######################################## # RUN
######################################## if [[ "$train" == "yes" ]]; then
 echo "[INFO] Running training..."
 python3 -u ../train.py "$CONFIG"
fi

if [[ "$validation" == "yes" ]]; then
 echo "[INFO] Running validation..."
 python3 -u ../eval.py "$CONFIG"
fi

if [[ "$train" != "yes" && "$validation" != "yes" ]]; then
 echo "[WARNING] Neither training nor validation selected."
fi

######################################## # END
######################################## end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"