"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""


import os
import glob
import yaml

base_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/vit/tests"
target_config_path = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/experiments/config_vit_all.yaml"

template_path = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/experiments/config_unet_experiments.yaml"
with open(template_path, 'r') as f:
    config = yaml.safe_load(f)

config['experiment'] = 'eval_vit_experiments'
datasets = []

datasets.append({
    'name': 'CNN_Base',
    'file_path': '../main/results/cnn/retained/cnn_exp5/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn/output_data/cnn_predictions_era5_to_mswep.nc',
    'variable_name': 'precipitation'
})

nc_files = glob.glob(f"{base_dir}/**/*.nc", recursive=True)
count = 0
for file in nc_files:
    if 'yaml_not_found' in file:
        continue
    if 'predictions_era5_to_mswep' not in file:
        continue
        
    model_name = "unknown"
    parts = file.split('/')
    for i, p in enumerate(parts):
        if p in ['tests_architecture', 'tests_advanced_arch', 'tests_hyperparams', 'tests_hybrids', 'tests_other']:
            model_name = parts[i+1]
            break
            
    datasets.append({
        'name': model_name,
        'file_path': file,
        'variable_name': 'precipitation'
    })
    count += 1

config['datasets'] = datasets
print(f"Added {count} ViT models.")

with open(target_config_path, 'w') as f:
    yaml.dump(config, f, sort_keys=False)

print(f"Created {target_config_path}")
