import os
import yaml
import stat

def generate_regional_setups():
    base_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling"
    main_dir = os.path.join(base_dir, "main")
    configs_dir = os.path.join(main_dir, "configs")
    scripts_dir = os.path.join(main_dir, "scripts")
    jobs_dir = os.path.join(scripts_dir, "jobs", "regional")
    logs_dir = os.path.join(scripts_dir, "logs", "regional")
    shapefiles_dir = os.path.join(base_dir, "data", "shape_files")
    
    models = ["cnn", "unet", "vit", "glm"]
    
    # Regions
    regions = {
        "north": os.path.join(shapefiles_dir, "north.shp"),
        "east": os.path.join(shapefiles_dir, "east.shp"),
        "south": os.path.join(shapefiles_dir, "south.shp")
    }
    
    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    for model in models:
        print(f"Processing model: {model}")
        base_yaml = os.path.join(configs_dir, model, "retained", "config.yaml")
        target_dir = os.path.join(configs_dir, model, "retained", "regional")
        os.makedirs(target_dir, exist_ok=True)
        
        if not os.path.exists(base_yaml):
            print(f"  [ERROR] Base config not found: {base_yaml}")
            continue
            
        with open(base_yaml, "r") as f:
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
            yaml_filename = f"config_{reg_name}.yaml"
            yaml_path = os.path.join(target_dir, yaml_filename)
            
            with open(yaml_path, "w") as out_f:
                yaml.dump(reg_config, out_f, default_flow_style=False)
                
            print(f"  -> Generated config: {yaml_path}")
            
            # Generate SLURM submission script
            slurm_script_path = os.path.join(jobs_dir, f"job_gpu_{model}_retained_{reg_name}.sh")
            
            # Generate script content
            content = f"""#!/bin/bash
#SBATCH --job-name={model}_ret_{reg_name}
#SBATCH --output={logs_dir}/{model}_ret_{reg_name}_%j.out
#SBATCH --error={logs_dir}/{model}_ret_{reg_name}_%j.err
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
echo "Model: {model.upper()} RETAINED (Region: {reg_name.upper()}) | Train: $train | Validation: $validation"
echo "======================================"

nvidia-smi

gpu_log="{logs_dir}/gpu_usage_{model}_ret_{reg_name}_${{SLURM_JOB_ID}}.log"
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
    generate_regional_setups()
