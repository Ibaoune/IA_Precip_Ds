#!/bin/bash
#SBATCH --job-name=postproc_retained
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=scripts/logs/postproc_retained_%j.log
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

echo "Starting SLURM Job: postproc_retained"
echo "Target Configuration: configs/config_retained.yaml"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Directory: $(pwd)"
echo "=========================================================="

# Activate Conda Environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Execute Master Post-Processing Runner
mkdir -p logs
python3 -u src/postproc.py configs/config_retained.yaml >> logs/postproc_retained_$(date +"%Y%m%d_%H%M%S").log 2>&1

echo "=========================================================="
echo "Job Completed Successfully!"
echo "Date: $(date)"
echo "=========================================================="
