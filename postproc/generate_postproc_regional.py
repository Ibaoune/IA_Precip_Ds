import os
import yaml
import copy

def generate_postproc_regional():
    base_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc"
    base_config_path = os.path.join(base_dir, "configs", "config_final_retained_models.yaml")
    out_config_path = os.path.join(base_dir, "configs", "config_retained_regional.yaml")
    
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)
        
    config["experiment"] = "Retained_Regional_Evaluation"
    
    new_datasets = []
    regions = ["north", "north_east", "south", "east"]
    
    config["parameters"]["regions"] = ["allmorr"] + regions
    
    # We only need DJF and JJA since Annual is already computed
    config["parameters"]["plot_periods"] = ["DJF", "JJA"]
    
    # Define mapping from global name to experiment name so we know how to replace
    exp_mapping = {
        "CNN": "cnn_exp5",
        "Unet": "unet_exp32_parallel",
        "Vit": "vit_precip_exp21_best_hybrid"
    }
    
    for ds in config["datasets"]:
        name = ds["name"]
        
        # Always include the global model
        if name in exp_mapping:
            ds_global = copy.deepcopy(ds)
            ds_global["name"] = f"{name}_Global"
            new_datasets.append(ds_global)
            
            # Now add regional variants
            base_exp = exp_mapping[name]
            for reg in regions:
                ds_reg = copy.deepcopy(ds)
                ds_reg["name"] = f"{name}_{reg.capitalize()}"
                # The training script appends _lossmask_region to the experiment name folder
                reg_exp = f"{base_exp}_lossmask_{reg}"
                ds_reg["file_path"] = ds_reg["file_path"].replace(f"/{base_exp}/", f"/{reg_exp}/")
                new_datasets.append(ds_reg)
        else:
            # Maybe GLM? Include it as is.
            new_datasets.append(ds)
            
    config["datasets"] = new_datasets
    
    # Update visualization colors for the new model names
    colors = {
        "MSWEP": "#000000",
        "GLM": "#2A9D8F",
        "CNN_Global": "#1D3557",
        "CNN_North": "#457B9D",
        "CNN_South": "#A8DADC",
        "CNN_East": "#8ECAE6",
        "Unet_Global": "#4A0082",
        "Unet_North": "#8A2BE2",
        "Unet_South": "#BBA0CA",
        "Unet_East": "#D8BFD8",
        "Vit_Global": "#9E0000",
        "Vit_North": "#E63946",
        "Vit_South": "#F4A261",
        "Vit_East": "#E9C46A"
    }
    config["visualisation"]["model_colors"] = colors
    
    with open(out_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
    print(f"Generated {out_config_path}")
    
    # Create the job script
    job_script_path = os.path.join(base_dir, "scripts", "jobs", "job_retained_regional.sh")
    job_content = f"""#!/bin/bash
#SBATCH --job-name=val_retained_regional
#SBATCH --output=postproc/logs/postproc_retained_regional_%j.log
#SBATCH --error=postproc/logs/postproc_retained_regional_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

source ~/.bashrc
conda activate clean_env_Pytorch

echo "=== Starting Post-Processing for Regional Models ==="
export PYTHONUNBUFFERED=1

cd postproc
CONFIG="configs/config_retained_regional.yaml"

python3 -u src/postproc.py "$CONFIG"
"""
    with open(job_script_path, "w") as f:
        f.write(job_content)
    
    os.chmod(job_script_path, 0o755)
    print(f"Generated {job_script_path}")

if __name__ == "__main__":
    generate_postproc_regional()
