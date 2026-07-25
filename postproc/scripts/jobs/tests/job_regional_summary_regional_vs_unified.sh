#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=reg_summary_comp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=scripts/logs/regional_summary_regional_vs_unified_%j.log
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

echo "Starting SLURM Job: regional_summary_regional_vs_unified"
echo "Target Configuration: configs/config_regional_vs_unified.yaml"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Directory: $(pwd)"
echo "=========================================================="

# Activate Conda Environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Generate regional heatmaps and barplots for all seasonal periods using shapefiles
mkdir -p logs
python3 -u src/regional_summary.py configs/config_regional_vs_unified.yaml --all-periods \
 >> logs/regional_summary_comp_$(date +"%Y%m%d_%H%M%S").log 2>&1

echo "=========================================================="
echo "Job Completed Successfully!"
echo "Date: $(date)"
echo "=========================================================="
