import os
import yaml

def generate_regional_setups():
    # Base paths
    base_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main"
    configs_dir = os.path.join(base_dir, "configs")
    scripts_dir = os.path.join(base_dir, "scripts")
    
    # Models and their configurations
    model_setups = [
        {
            "model_type": "cnn",
            "name": "cnn_exp3",
            "base_yaml": os.path.join(configs_dir, "cnn", "cnn_exp3.yaml"),
            "target_dir": os.path.join(configs_dir, "cnn", "regional"),
            "gpu": True
        },
        {
            "model_type": "cnn",
            "name": "cnn_exp5",
            "base_yaml": os.path.join(configs_dir, "cnn", "cnn_exp5.yaml"),
            "target_dir": os.path.join(configs_dir, "cnn", "regional"),
            "gpu": True
        },
        {
            "model_type": "vit",
            "name": "vit",
            "base_yaml": os.path.join(configs_dir, "vit", "config.yaml"),
            "target_dir": os.path.join(configs_dir, "vit", "regional"),
            "gpu": True
        },
        {
            "model_type": "glm",
            "name": "glm",
            "base_yaml": os.path.join(configs_dir, "glm", "config.yaml"),
            "target_dir": os.path.join(configs_dir, "glm", "regional"),
            "gpu": False
        }
    ]
    
    # Regional split specifications
    regions = {
        "north": {
            "lat_min": 28,
            "lat_max_override": None, # Will use original lat_max
            "suffix": "_north"
        },
        "south": {
            "lat_min": 21,
            "lat_max_override": 28,
            "suffix": "_south"
        }
    }
    
    # Ensure targets exist
    os.makedirs(os.path.join(scripts_dir, "regional"), exist_ok=True)
    
    for setup in model_setups:
        print(f"Processing model: {setup['name']} from {setup['base_yaml']}")
        os.makedirs(setup["target_dir"], exist_ok=True)
        
        # Load base configuration
        with open(setup["base_yaml"], "r") as f:
            base_config = yaml.safe_load(f)
            
        original_lat_max = base_config["region"]["lat_max"]
        original_experiment = base_config["general"]["experiment"]
        
        for reg_name, reg_spec in regions.items():
            reg_config = yaml.safe_load(yaml.dump(base_config)) # Deep copy
            
            # Apply regional overrides
            reg_config["general"]["experiment"] = original_experiment + reg_spec["suffix"]
            reg_config["region"]["lat_min"] = reg_spec["lat_min"]
            
            if reg_spec["lat_max_override"] is not None:
                reg_config["region"]["lat_max"] = reg_spec["lat_max_override"]
            else:
                reg_config["region"]["lat_max"] = original_lat_max
                
            # Define output filename
            yaml_filename = f"{setup['name']}_{reg_name}.yaml" if setup['model_type'] == "cnn" else f"{reg_name}.yaml"
            yaml_path = os.path.join(setup["target_dir"], yaml_filename)
            
            with open(yaml_path, "w") as out_f:
                yaml.dump(reg_config, out_f, default_flow_style=False)
                
            print(f"  -> Generated config: {yaml_path}")
            
            # Generate SLURM submission script
            relative_yaml_path = os.path.relpath(yaml_path, os.path.join(scripts_dir, "regional"))
            slurm_script_path = os.path.join(scripts_dir, "regional", f"job_{'gpu' if setup['gpu'] else 'cpu'}_{setup['name']}_{reg_name}.sh")
            
            # Generate script content
            if setup["gpu"]:
                content = f"""#!/bin/bash

#SBATCH --job-name={setup['name']}_{reg_name}_gpu
#SBATCH --output={setup['name']}_{reg_name}_gpu%j.log
#SBATCH --error={setup['name']}_{reg_name}_gpu%j.log
#SBATCH --partition=gpu
#SBATCH --qos=default-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-7KSIFKVWKUY-DEFAULT-GPU

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
echo "Model: {setup['name']} ({reg_name.upper()}) | Train: $train | Validation: $validation"
echo "======================================"

nvidia-smi

gpu_log="gpu_usage_{setup['name']}_{reg_name}_${{SLURM_JOB_ID}}.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \\
           --format=csv,nounits,noheader \\
           --loop=60 > "$gpu_log" &

CONFIG="{relative_yaml_path}"

if [[ "$train" == "yes" ]]; then
    echo "[INFO] Running training..."
    python3 -u ../../train.py "$CONFIG"
fi

if [[ "$validation" == "yes" ]]; then
    echo "[INFO] Running validation..."
    python3 -u ../../eval.py "$CONFIG"
fi

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${{SLURM_JOB_ID}} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"
"""
            else:
                content = f"""#!/bin/bash

#SBATCH --job-name={setup['name']}_{reg_name}_cpu
#SBATCH --output={setup['name']}_{reg_name}_cpu%j.log
#SBATCH --error={setup['name']}_{reg_name}_cpu%j.log
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
echo "Model: {setup['name']} ({reg_name.upper()}) | Train: $train | Validation: $validation"
echo "======================================"

CONFIG="{relative_yaml_path}"

if [[ "$train" == "yes" ]]; then
    echo "[INFO] Running training..."
    python3 -u ../../train.py "$CONFIG"
fi

if [[ "$validation" == "yes" ]]; then
    echo "[INFO] Running validation..."
    python3 -u ../../eval.py "$CONFIG"
fi

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "======================================"
echo "Job ${{SLURM_JOB_ID}} completed in $runtime seconds."
echo "End time: $(date)"
echo "======================================"
"""
            
            with open(slurm_script_path, "w") as out_sh:
                out_sh.write(content)
                
            os.chmod(slurm_script_path, 0o755)
            print(f"  -> Generated SLURM script: {slurm_script_path}")

if __name__ == "__main__":
    generate_regional_setups()
