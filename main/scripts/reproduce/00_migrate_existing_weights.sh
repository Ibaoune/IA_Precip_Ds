#!/bin/bash
# ==============================================================================
# Script: 00_migrate_existing_weights.sh
# Description: Copies the pre-trained weights from the development/testing 
#              directories into the isolated 'reproduce' directories so you 
#              can run the evaluation scripts (03, 04, 05) immediately without 
#              waiting for 24h of retraining (scripts 01 and 02).
# ==============================================================================

set -e

# Change to the root of the project
cd "$(dirname "$0")/../../.."

echo "=== Migrating pre-trained weights to reproduce directories ==="

# Define models and their original output folders
declare -A GLOBAL_PATHS=(
    ["cnn"]="cnn/tests/tests_advanced_arch/cnn_exp5/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn"
    ["unet"]="unet/tests/tests_advanced_arch/unet_exp32_parallel/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn"
    ["vit"]="vit/retained/vit_precip_exp21_best_hybrid/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn"
    ["glm"]="glm/retained/glm_precip_l2/region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/test_2006_01_01_2020_12_31/gridbox_0.001_bernoulli_gamma_10ep_wd"
)

REGIONS=("north" "north_east" "south" "east")

for model in "${!GLOBAL_PATHS[@]}"; do
    base_path="main/results/${GLOBAL_PATHS[$model]}"
    
    # Check if original base path exists
    if [ ! -d "$base_path" ]; then
        echo "[WARNING] Base path not found: $base_path"
        continue
    fi

    # 1. Copy Global Model
    target_global="main/results/reproduce/global/${model}"
    echo "Copying global model for $model..."
    mkdir -p "$target_global"
    rsync -av --exclude="output_data" "$base_path/" "$target_global/" > /dev/null

    # 2. Copy Regional Models
    echo "Copying regional models for $model..."
    # We need to replace the experiment name in the path with the regional variant.
    # e.g., cnn_exp5 -> cnn_exp5_lossmask_north
    # We can extract the experiment name by looking at the folder just before /region_lat...
    exp_name=$(echo "$base_path" | grep -oP '(?<=/)[^/]+(?=/region_lat)')

    for reg in "${REGIONS[@]}"; do
        reg_exp="${exp_name}_lossmask_${reg}"
        reg_path="${base_path/$exp_name/$reg_exp}"

        target_reg="main/results/reproduce/regional/${model}_${reg}"
        
        if [ -d "$reg_path" ]; then
            mkdir -p "$target_reg"
            rsync -av --exclude="output_data" "$reg_path/" "$target_reg/" > /dev/null
        else
            echo "  [WARNING] Regional path not found: $reg_path"
        fi
    done
done

echo "=== Migration Complete ==="
echo "You can now safely run scripts 03, 04, and 05."
