# Author: M. El Aabaribaoune (@um6p)
import os
import yaml

def generate_lossmask_setups():
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
            "target_dir": os.path.join(configs_dir, "cnn", "loss_mask"),
            "gpu": True
        },
        {
            "model_type": "cnn",
            "name": "cnn_exp5",
            "base_yaml": os.path.join(configs_dir, "cnn", "cnn_exp5.yaml"),
            "target_dir": os.path.join(configs_dir, "cnn", "loss_mask"),
            "gpu": True
        },
        {
            "model_type": "vit",
            "name": "vit",
            "base_yaml": os.path.join(configs_dir, "vit", "config.yaml"),
            "target_dir": os.path.join(configs_dir, "vit", "loss_mask"),
            "gpu": True
        }
    ]
    
    # Sub-domain specifications
    regions = {
        "north": os.path.join(shapefiles_dir, "north.shp"),
        "north_east": os.path.join(shapefiles_dir, "north_east.shp"),
        "east": os.path.join(shapefiles_dir, "east.shp"),
        "south": os.path.join(shapefiles_dir, "south.shp")
    }
    
    # Ensure targets exist
    os.makedirs(os.path.join(scripts_dir, "loss_mask"), exist_ok=True)
    
    for setup in model_setups:
        print(f"Processing model: {setup['name']} from {setup['base_yaml']}")
        os.makedirs(setup["target_dir"], exist_ok=True)
        
        # Load base configuration
        with open(setup["base_yaml"], "r") as f:
            base_config = yaml.safe_load(f)
        
        for reg_name, shape_path in regions.items():
            reg_config = yaml.safe_load(yaml.dump(base_config)) # Deep copy
            
            # Inject loss mask config under training
            if "training" not in reg_config:
                reg_config["training"] = {}
            
            reg_config["training"]["loss_mask"] = {
                "enable": True,
                "region": reg_name,
                "shapefile": shape_path
            }
            
            # Define output filename
            yaml_filename = f"{setup['name']}_lossmask_{reg_name}.yaml"
            yaml_path = os.path.join(setup["target_dir"], yaml_filename)
            
            with open(yaml_path, "w") as out_f:
                yaml.dump(reg_config, out_f, default_flow_style=False)
                
            print(f"  -> Generated config: {yaml_path}")
            
            # Generate SLURM submission script
            relative_yaml_path = os.path.relpath(yaml_path, os.path.join(scripts_dir, "loss_mask"))
            slurm_script_path = os.path.join(scripts_dir, "loss_mask", f"job_gpu_{setup['name']}_lossmask_{reg_name}.sh")
            
            # Generate script content
            content = f"""#!/bin/bash
#SBATCH --job-name={setup['name']}_lm_{reg_name}
#SBATCH --output={setup['name']}_lm_{reg_name}_%j.log
#SBATCH --error={setup['name']}_lm_{reg_name}_%j.log
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
echo "Model: {setup['name']} (Loss Mask: {reg_name.upper()}) | Train: $train | Validation: $validation"
echo "======================================"

nvidia-smi

gpu_log="gpu_usage_{setup['name']}_lm_{reg_name}_${{SLURM_JOB_ID}}.log"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \\
           --format=csv,nounits,noheader \\
           --loop=60 > "$gpu_log" &

CONFIG="{yaml_path}"

cd {main_dir}

if [[ "$train" == "yes" ]]; then
    echo "[INFO] Running training..."
    python3 -u train.py "$CONFIG"
fi

if [[ "$validation" == "yes" ]]; then
    echo "[INFO] Running validation..."
    python3 -u eval.py "$CONFIG"
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
    generate_lossmask_setups()
