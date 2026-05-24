#!/bin/bash
#SBATCH --job-name=pp_regional_vs_unified
#SBATCH --output=scripts/logs/postproc_regional_vs_unified_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
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

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Post-processing Regional vs Unified Comparison"
echo "Directory: $(pwd)"
echo "======================================"

echo "Starting regional comparison post-processing suite."
python3 -u src/postproc.py configs/config_regional_vs_unified.yaml

echo "======================================"
echo "Job completed."
echo "======================================"
