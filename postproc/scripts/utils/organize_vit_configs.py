"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""


import os
import shutil
import yaml
import glob

configs_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/configs/vit/tests"
results_dir = "/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/vit/tests"

groups = {
    'tests_architecture': ['test_default', 'test_exp1', 'test_exp2', 'test_exp3', 'test_exp4', 'test_exp10', 'test_exp12', 'test_exp14', 'test_exp16', 'test_exp18', 'test_exp20'],
    'tests_hyperparams': ['test_exp5', 'test_exp6', 'test_exp7', 'test_exp8', 'test_exp9', 'test_exp15', 'test_exp17'],
    'tests_advanced_arch': ['test_exp11', 'test_exp13', 'test_exp19'],
    'tests_hybrids': ['test_exp21_best_hybrid', 'test_exp21_hybrid_base', 'test_exp22_hybrid_deep_reg', 'test_exp23_hybrid_bilinear_channel', 'test_exp24_hybrid_wide_global', 'test_exp25_hybrid_fast_cosine'],
    'tests_other': ['config_old', 'test']
}

# Move everything out of yaml_not_found first
yaml_not_found = os.path.join(results_dir, 'yaml_not_found')
if os.path.exists(yaml_not_found):
    for item in os.listdir(yaml_not_found):
        src = os.path.join(yaml_not_found, item)
        dst = os.path.join(results_dir, item)
        if not os.path.exists(dst):
            shutil.move(src, dst)

os.makedirs(yaml_not_found, exist_ok=True)

exp_to_group = {}
for folder, configs in groups.items():
    for cfg in configs:
        dst = os.path.join(configs_dir, folder, f"{cfg}.yaml")
        if os.path.exists(dst):
            with open(dst, 'r') as f:
                cfg_data = yaml.safe_load(f)
                if cfg_data and 'general' in cfg_data and 'experiment' in cfg_data['general']:
                    exp_name = cfg_data['general']['experiment']
                    exp_to_group[exp_name] = folder

print(f"Mapped {len(exp_to_group)} experiments from YAMLs.")

for item in os.listdir(results_dir):
    item_path = os.path.join(results_dir, item)
    if not os.path.isdir(item_path): continue
    if item in groups.keys() or item == 'yaml_not_found' or item.startswith('tests_') or item == 'region_training' or item == 'vit_precip_test_mask': continue
    
    if item in exp_to_group:
        target_group = exp_to_group[item]
        nc_files = glob.glob(f"{item_path}/**/*.nc", recursive=True)
        if len(nc_files) > 0:
            shutil.move(item_path, os.path.join(results_dir, target_group, item))
        else:
            shutil.move(item_path, os.path.join(yaml_not_found, item))
    else:
        shutil.move(item_path, os.path.join(yaml_not_found, item))

print("Organization Complete!")
