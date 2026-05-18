#!/bin/bash
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

echo "=== Starting In-Situ Weather Station Post-Processing ==="

echo "[STEP 1/4] Running Climatology & Taylor polar plotting..."
python3 -u src/insitu_climatology.py --config config.yaml

echo "[STEP 2/4] Running Quantile-Quantile plotting..."
python3 -u src/insitu_qqplot.py --config config.yaml

echo "[STEP 3/4] Running Taylor Diagram plotting..."
python3 -u src/taylor_insitu.py --config config.yaml

echo "[STEP 4/4] Running Annual Cycle with RMSE & Bias calculations..."
python3 -u src/annual_cycle.py --config config.yaml

echo "=== All In-Situ Post-Processing Scripts Completed! ==="
