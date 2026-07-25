#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=seasonal_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=scripts/logs/seasonal_summary_%j.log
#SBATCH --time=01:00:00
#SBATCH --mem=32G
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

echo "Starting SLURM Job: seasonal_summary"
echo "Target Configuration: configs/config_retained.yaml"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Directory: $(pwd)"
echo "=========================================================="

# Activate Conda Environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Generate seasonal heatmaps and barplots (all periods in one run)
mkdir -p logs
python3 -u src/seasonal_summary.py configs/config_retained.yaml \
 >> logs/seasonal_summary_$(date +"%Y%m%d_%H%M%S").log 2>&1

echo "=========================================================="
echo "Job Completed Successfully!"
echo "Date: $(date)"
echo "=========================================================="
