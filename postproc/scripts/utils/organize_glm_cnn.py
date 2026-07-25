"""
Author: M. El Aabaribaoune (@um6p)
Description: Utility functions for data processing, statistical analysis, and plotting.
"""


import os
import shutil
import yaml
import glob

def organize_model(model_name, groups, readmes):
    print(f"--- Organizing {model_name.upper()} ---")
    configs_dir = f"/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/configs/{model_name}/tests"
    results_dir = f"/srv/data/mohammad.elaabaribao/work/papers/downscaling/main/results/{model_name}/tests"
    
    yaml_not_found = os.path.join(results_dir, 'yaml_not_found')
    
    # Create subdirectories in configs and add readmes
    for folder in groups.keys():
        os.makedirs(os.path.join(configs_dir, folder), exist_ok=True)
        if folder in readmes:
            with open(os.path.join(configs_dir, folder, 'README.md'), 'w') as f:
                f.write(readmes[folder])
    
    # Move everything out of yaml_not_found first if it exists
    if os.path.exists(yaml_not_found):
        for item in os.listdir(yaml_not_found):
            src = os.path.join(yaml_not_found, item)
            dst = os.path.join(results_dir, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
    os.makedirs(yaml_not_found, exist_ok=True)

    exp_to_group = {}
    
    # Parse YAML files and move them
    for folder, configs in groups.items():
        for cfg in configs:
            src = os.path.join(configs_dir, f"{cfg}.yaml")
            dst = os.path.join(configs_dir, folder, f"{cfg}.yaml")
            if os.path.exists(src):
                shutil.move(src, dst)
            if os.path.exists(dst):
                with open(dst, 'r') as f:
                    cfg_data = yaml.safe_load(f)
                    if cfg_data and 'general' in cfg_data and 'experiment' in cfg_data['general']:
                        exp_name = cfg_data['general']['experiment']
                        exp_to_group[exp_name] = folder

    print(f"Mapped {len(exp_to_group)} {model_name} experiments from YAMLs.")

    # Organize Results
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if not os.path.isdir(item_path): continue
        if item in groups.keys() or item == 'yaml_not_found' or item.startswith('tests_') or item == 'region_training': continue
        
        if item in exp_to_group:
            target_group = exp_to_group[item]
            nc_files = glob.glob(f"{item_path}/**/*.nc", recursive=True)
            # Make sure destination folder exists
            os.makedirs(os.path.join(results_dir, target_group), exist_ok=True)
            if len(nc_files) > 0:
                shutil.move(item_path, os.path.join(results_dir, target_group, item))
            else:
                shutil.move(item_path, os.path.join(yaml_not_found, item))
        else:
            shutil.move(item_path, os.path.join(yaml_not_found, item))


# GLM SETUP
glm_groups = {
    'tests_regularization': ['alpha_l1', 'alpha_l2'],
    'tests_interpolation': ['interp_bilinear', 'interp_nearest'],
    'tests_other': ['config_old', 'test']
}
glm_readmes = {
    'tests_regularization': "# Tests de Régularisation GLM\n\nComparaison des effets des régularisations Lasso (L1) et Ridge (L2) sur le GLM.",
    'tests_interpolation': "# Tests d'Interpolation GLM\n\nImpact du mode d'interpolation spatiale initiale des variables prédictrices (bilinéaire vs plus proche voisin).",
    'tests_other': "# Tests Divers GLM\n\nAnciennes itérations et tests temporaires."
}

# CNN SETUP
cnn_groups = {
    'tests_architecture': ['cnn_exp3'],
    'tests_training': ['gridbox_LR_scheduler', 'gridbox_scheduler', 'gridbox_epochs'],
    'tests_regularization': ['gridbox_dropout', 'gridbox_weight_decay', 'gridbox_gradient_clipping', 'gridbox_group_norm'],
    'tests_other': ['config_old', 'test']
}
cnn_readmes = {
    'tests_architecture': "# Tests d'Architecture CNN\n\nConfigurations exploratoires de base pour l'architecture CNN.",
    'tests_training': "# Tests d'Entraînement CNN\n\nOptimisations du taux d'apprentissage, schedulers et nombre d'époques.",
    'tests_regularization': "# Tests de Régularisation CNN\n\nÉtudes sur le Dropout, le Weight Decay, le Gradient Clipping et le Group Normalization.",
    'tests_other': "# Tests Divers CNN\n\nAnciennes itérations et tests temporaires."
}

organize_model("glm", glm_groups, glm_readmes)
organize_model("cnn", cnn_groups, cnn_readmes)

print("Organization Complete for GLM and CNN!")
