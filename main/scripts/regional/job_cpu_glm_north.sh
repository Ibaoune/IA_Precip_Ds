#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)

#SBATCH --job-name=glm_north_cpu
#SBATCH --output=glm_north_cpu%j.log
#SBATCH --error=glm_north_cpu%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

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
echo "Model: glm (NORTH) | Train: $train | Validation: $validation"
echo "======================================"

CONFIG="/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/configs/glm/regional/north.yaml"

cd /srv/data/mohammad.elaabaribao/work/papers/downscaling/main

if [[ "$train" == "yes" ]]; then
    echo "[INFO] Running training..."
    python3 -u train.py "$CONFIG"
fi

if [[ "$validation" == "yes" ]]; then
    echo "[INFO] Running validation..."
    python3 -u eval.py "$CONFIG"
fi

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"
