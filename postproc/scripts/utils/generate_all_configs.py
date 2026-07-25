"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""


import os
import glob
import yaml

def generate_config(model_name):
    print(f"--- Generating {model_name.upper()} Config ---")
    base_dir = f"/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/{model_name}/tests"
    target_config_path = f"/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/experiments/config_{model_name}_all.yaml"

    template_path = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/experiments/config_unet_experiments.yaml"
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    config['experiment'] = f'eval_{model_name}_experiments'
    config['postproc']['mean']['rmse'] = False
    config['postproc']['mean']['correlation'] = False
    config['postproc']['extreme']['enable'] = False
    
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
            
        m_name = "unknown"
        parts = file.split('/')
        for i, p in enumerate(parts):
            if p.startswith('tests_'):
                m_name = parts[i+1]
                break
                
        datasets.append({
            'name': m_name,
            'file_path': file,
            'variable_name': 'precipitation'
        })
        count += 1

    config['datasets'] = datasets
    print(f"Added {count} {model_name.upper()} models.")

    with open(target_config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

generate_config('unet')
generate_config('vit')
generate_config('cnn')
generate_config('glm')
print("Configurations updated successfully!")
