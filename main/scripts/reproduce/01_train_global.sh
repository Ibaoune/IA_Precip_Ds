#!/bin/bash
# ==============================================================================
# Script: 01_train_global.sh
# Description: Submits the training jobs for the 4 retained global models 
#              (CNN, U-Net, ViT, GLM) over the entire Moroccan domain.
# ==============================================================================

set -e

# Change to the root of the project
cd "$(dirname "$0")/../../.."

echo "=== Submitting Global Training Jobs ==="

MODELS=("cnn" "unet" "vit" "glm")
BASE_CONFIGS_DIR="main/configs/reproduce/base"
GEN_CONFIGS_DIR="main/configs/reproduce/global"
JOBS_DIR="main/scripts/jobs/reproduce/global"
LOGS_DIR="main/scripts/logs/reproduce/global"

mkdir -p "$GEN_CONFIGS_DIR" "$JOBS_DIR" "$LOGS_DIR"

cat << 'EOF' > main/scripts/reproduce/generate_global.py
import os
import yaml

def main():
    models = ["cnn", "unet", "vit", "glm"]
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    
    for model in models:
        base_yaml = os.path.join(base_dir, "main", "configs", "reproduce", "base", f"config_{model}.yaml")
        out_yaml = os.path.join(base_dir, "main", "configs", "reproduce", "global", f"config_{model}.yaml")
        
        if not os.path.exists(base_yaml):
            continue
            
        with open(base_yaml, "r") as f:
            config = yaml.safe_load(f)
            
        # Reroute results to isolated reproduce directory
        config["paths"]["results_dir"] = os.path.join(base_dir, "main", "results", "reproduce", "global", model)
        
        with open(out_yaml, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    main()
EOF

python3 main/scripts/reproduce/generate_global.py

for model in "${MODELS[@]}"; do
    CONFIG_PATH="${GEN_CONFIGS_DIR}/config_${model}.yaml"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "[WARNING] Config not generated for ${model}: ${CONFIG_PATH}"
        continue
    fi

    JOB_SCRIPT="${JOBS_DIR}/job_gpu_${model}_global.sh"

    cat << 'EOF2' > "$JOB_SCRIPT"
#!/bin/bash
#SBATCH --job-name=__MODEL___global
#SBATCH --output=__LOGS_DIR__/__MODEL___global_%j.out
#SBATCH --error=__LOGS_DIR__/__MODEL___global_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

source ~/.bashrc
conda activate clean_env_Pytorch
export PYTHONUNBUFFERED=1

cd main
echo "Starting global training for __MODEL__"
python3 -u train.py ../__CONFIG_PATH__
python3 -u eval.py ../__CONFIG_PATH__
EOF2

    sed -i "s|__MODEL__|${model}|g" "$JOB_SCRIPT"
    sed -i "s|__LOGS_DIR__|${LOGS_DIR}|g" "$JOB_SCRIPT"
    sed -i "s|__CONFIG_PATH__|${CONFIG_PATH}|g" "$JOB_SCRIPT"

    chmod +x "$JOB_SCRIPT"
    echo "Submitting $JOB_SCRIPT"
    sbatch "$JOB_SCRIPT"
done

echo "=== Global Training Jobs Submitted ==="
