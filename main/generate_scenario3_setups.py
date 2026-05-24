import os
import yaml

def generate_scenario3_setups():
    # Base paths
    base_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling"
    main_dir = os.path.join(base_dir, "main")
    configs_dir = os.path.join(main_dir, "configs")
    scripts_dir = os.path.join(main_dir, "scripts")
    postproc_dir = os.path.join(base_dir, "postproc")
    shapefiles_dir = os.path.join(postproc_dir, "shape_files")
    
    # Models and their configurations
    model_setups = [
        {
            "model_type": "cnn",
            "name": "cnn_exp3",
            "base_yaml": os.path.join(configs_dir, "cnn", "cnn_exp3.yaml"),
            "target_dir": os.path.join(configs_dir, "cnn", "scenario3"),
            "experiment_prefix": "cnnexp3_scenario3_",
            "gpu": True
        },
        {
            "model_type": "cnn",
            "name": "cnn_exp5",
            "base_yaml": os.path.join(configs_dir, "cnn", "cnn_exp5.yaml"),
            "target_dir": os.path.join(configs_dir, "cnn", "scenario3"),
            "experiment_prefix": "cnnexp5_scenario3_",
            "gpu": True
        },
        {
            "model_type": "vit",
            "name": "vit",
            "base_yaml": os.path.join(configs_dir, "vit", "config.yaml"),
            "target_dir": os.path.join(configs_dir, "vit", "scenario3"),
            "experiment_prefix": "vit_scenario3_",
            "gpu": True
        }
    ]
    
    # Macro-region specifications
    regions = {
        "north_northeast": [
            os.path.join(shapefiles_dir, "north.shp"),
            os.path.join(shapefiles_dir, "north_east.shp")
        ],
        "east_south": [
            os.path.join(shapefiles_dir, "east.shp"),
            os.path.join(shapefiles_dir, "south.shp")
        ]
    }
    
    # Ensure target script directory exists
    os.makedirs(os.path.join(scripts_dir, "scenario3"), exist_ok=True)
    
    for setup in model_setups:
        print(f"Processing model: {setup['name']} from {setup['base_yaml']}")
        os.makedirs(setup["target_dir"], exist_ok=True)
        
        # Load base configuration
        with open(setup["base_yaml"], "r") as f:
            base_config = yaml.safe_load(f)
        
        for reg_name, shape_paths in regions.items():
            reg_config = yaml.safe_load(yaml.dump(base_config)) # Deep copy
            
            # Update experiment name inside the config to exactly what user wants
            experiment_name = f"{setup['experiment_prefix']}{reg_name}"
            reg_config["general"]["experiment"] = experiment_name
            
            # Inject loss mask config under training
            if "training" not in reg_config:
                reg_config["training"] = {}
            
            reg_config["training"]["loss_mask"] = {
                "enable": True,
                "region": reg_name,
                "shapefile": shape_paths
            }
            
            # Define output filename
            # e.g., cnn_exp3_scenario3_north_northeast.yaml or vit_scenario3_north_northeast.yaml
            yaml_filename = f"{setup['name']}_scenario3_{reg_name}.yaml"
            yaml_path = os.path.join(setup["target_dir"], yaml_filename)
            
            with open(yaml_path, "w") as out_f:
                yaml.dump(reg_config, out_f, default_flow_style=False)
                
            print(f"  -> Generated config: {yaml_path}")
            
            # Generate SLURM submission script
            slurm_script_path = os.path.join(scripts_dir, "scenario3", f"job_gpu_{setup['name']}_scenario3_{reg_name}.sh")
            
            # Generate script content
            content = f"""#!/bin/bash
#SBATCH --job-name={experiment_name}
#SBATCH --output={experiment_name}_%j.log
#SBATCH --error={experiment_name}_%j.log
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
echo "Model: {setup['name']} (Macro Region: {reg_name.upper()}) | Train: $train | Validation: $validation"
echo "======================================"

nvidia-smi

gpu_log="gpu_usage_{experiment_name}_${{SLURM_JOB_ID}}.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \\
           --format=csv,nounits,noheader \\
           --loop=60 > "$gpu_log" &

CONFIG="{yaml_path}"

if [[ "$train" == "yes" ]]; then
    echo "[INFO] Running training..."
    python3 -u "{os.path.join(main_dir, 'train.py')}" "$CONFIG"
fi

if [[ "$validation" == "yes" ]]; then
    echo "[INFO] Running validation..."
    python3 -u "{os.path.join(main_dir, 'eval.py')}" "$CONFIG"
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
    generate_scenario3_setups()
