#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=insitu_plot
#SBATCH --output=out_insitu_%j.log
#SBATCH --error=out_insitu_%j.log
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1
export PYTHONPATH=src:$PYTHONPATH

# Ensure logs directory exists
mkdir -p logs

echo "=== Starting In-Situ Weather Station Post-Processing (Parallel Execution) ==="

echo "[STEP 1/4] Starting Climatology & Taylor polar plotting in background..."
python3 -u src/insitu_climatology.py --config config.yaml > logs/insitu_climatology.log 2>&1 &

echo "[STEP 2/4] Starting Quantile-Quantile plotting in background..."
python3 -u src/insitu_qqplot.py --config config.yaml > logs/insitu_qqplot.log 2>&1 &

echo "[STEP 3/4] Starting Taylor Diagram plotting in background..."
python3 -u src/taylor_insitu.py --config config.yaml > logs/taylor_insitu.log 2>&1 &

echo "[STEP 4/4] Starting Annual Cycle with RMSE & Bias calculations in background..."
python3 -u src/annual_cycle.py --config config.yaml > logs/annual_cycle.log 2>&1 &

echo "Waiting for all parallel post-processing steps to complete..."
wait

echo "=== All In-Situ Post-Processing Scripts Completed! ==="
