"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""


import os
import glob
import yaml

base_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/unet/tests"
target_config_path = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/experiments/config_unet_all.yaml"

# Load template from config_unet_experiments.yaml
template_path = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/experiments/config_unet_experiments.yaml"
with open(template_path, 'r') as f:
    config = yaml.safe_load(f)

# Reset datasets
datasets = []

# Add CNN_Base explicitly
datasets.append({
    'name': 'CNN_Base',
    'file_path': '../main/results/cnn/retained/cnn_exp5/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn/output_data/cnn_predictions_era5_to_mswep.nc',
    'variable_name': 'precipitation'
})

# Scan for all models in main/results/unet/tests/
nc_files = glob.glob(f"{base_dir}/**/*.nc", recursive=True)
count = 0
for file in nc_files:
    if 'yaml_not_found' in file:
        continue
    if 'predictions_era5_to_mswep' not in file:
        continue
        
    # Get model name from the directory name
    parts = file.split('/')
    # The directory name is usually 3 levels up from the nc file? 
    # e.g., tests_advanced_arch/unet_doury/.../output_data/unet_doury_predictions.nc
    # Let's find the folder under tests_advanced_arch, tests_architecture, etc.
    for i, p in enumerate(parts):
        if p in ['tests_architecture', 'tests_advanced_arch', 'tests_hyperparams', 'tests_predictors']:
            model_name = parts[i+1]
            break
            
    # Fix absolute path to relative if necessary, or just use absolute
    datasets.append({
        'name': model_name,
        'file_path': file,
        'variable_name': 'precipitation'
    })
    count += 1

config['datasets'] = datasets
print(f"Added {count} U-Net models.")

# Write new config
with open(target_config_path, 'w') as f:
    yaml.dump(config, f, sort_keys=False)

print(f"Created {target_config_path}")
