#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)

#SBATCH --job-name=cnn_exp5_gpu
#SBATCH --output=cnn_exp5_gpu%j.log
#SBATCH --error=cnn_exp5_gpu%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

train="yes"
validation="yes"

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Ensure Python logs are flushed immediately
export PYTHONUNBUFFERED=1

start_time=$(date +%s)

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Model: CNN (Exp 5) | Train: $train | Validation: $validation"
echo "======================================"

# GPU Info
nvidia-smi

gpu_log="gpu_usage_cnn_exp5_${SLURM_JOB_ID}.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
 --format=csv,nounits,noheader \
 --loop=60 > "$gpu_log" &

CONFIG="../configs/cnn/cnn_exp5.yaml"

if [[ "$train" == "yes" ]]; then
 echo "[INFO] Running training..."
 python3 -u ../train.py "$CONFIG"
fi

if [[ "$validation" == "yes" ]]; then
 echo "[INFO] Running validation..."
 python3 -u ../eval.py "$CONFIG"
fi

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"
