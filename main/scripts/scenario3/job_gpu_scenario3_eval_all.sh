#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=scenario3_eval_all
#SBATCH --output=scenario3_eval_all_%j.log
#SBATCH --error=scenario3_eval_all_%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

start_time=$(date +%s)

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running evaluation for all 6 Scenario 3 models"
echo "======================================"

nvidia-smi

BASE_DIR="/srv/data/mohammad.elaabaribao/work/papers/downscaling/main"

CONFIGS=(
  "configs/cnn/scenario3/cnn_exp3_scenario3_east_south.yaml"
  "configs/cnn/scenario3/cnn_exp3_scenario3_north_northeast.yaml"
  "configs/cnn/scenario3/cnn_exp5_scenario3_east_south.yaml"
  "configs/cnn/scenario3/cnn_exp5_scenario3_north_northeast.yaml"
  "configs/vit/scenario3/vit_scenario3_east_south.yaml"
  "configs/vit/scenario3/vit_scenario3_north_northeast.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "--------------------------------------------------"
  echo "[INFO] Evaluating config: $config"
  echo "--------------------------------------------------"
  python3 -u "$BASE_DIR/eval.py" "$BASE_DIR/$config"
done

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "All evaluations completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"
