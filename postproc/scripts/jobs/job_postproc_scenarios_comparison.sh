#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=postproc_scenarios_comparison
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=scripts/logs/postproc_scenarios_comparison_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

# Find repository root starting from SLURM_SUBMIT_DIR or current directory
SEARCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
while [ "$SEARCH_DIR" != "/" ]; do
    if [ -f "$SEARCH_DIR/src/postproc.py" ]; then
        cd "$SEARCH_DIR"
        break
    fi
    SEARCH_DIR=$(dirname "$SEARCH_DIR")
done

echo "Starting SLURM Job: postproc_scenarios_comparison"
echo "Target Configuration: configs/config_scenarios_comparison.yaml"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Directory: $(pwd)"
echo "=========================================================="

# Activate Conda Environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Execute Master Post-Processing Runner
mkdir -p logs
echo "[INFO] Running main post-processing metrics pipeline..."
python3 -u src/postproc.py configs/config_scenarios_comparison.yaml

echo ""
echo "[INFO] Running regional summary script (heatmaps, bar plots, CSVs for sub-regions)..."
python3 -u src/regional_summary.py configs/config_scenarios_comparison.yaml --all-periods

echo ""
echo "[INFO] Running seasonal summary script (heatmaps, bar plots, CSVs for seasons)..."
python3 -u src/seasonal_summary.py configs/config_scenarios_comparison.yaml

echo "=========================================================="
echo "Job Completed!"
echo "Date: $(date)"
echo "=========================================================="
