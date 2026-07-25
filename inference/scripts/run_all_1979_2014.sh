#!/bin/bash
# ==============================================================================
# Author: M. El Aabaribaoune (@um6p)
# Description: Job submission script for running inference on the SLURM cluster.
# ==============================================================================

#SBATCH --job-name=GLMinfer_1979_2014
#SBATCH --output=logs/out_infer_1979_2014_%j.log
#SBATCH --error=logs/out_infer_1979_2014_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "=================================================="
echo "Starting Multi-Model Inference in PARALLEL (1979-2014)"
echo "=================================================="

echo "Launching 1. GLM Precip L2..."
python3 -u src/predict.py tests/config_glm_1979_2014.yaml > logs/glm_1979_2014.log 2>&1 &
 
# echo "Launching 2. CNN Exp 5..."
# python3 -u src/predict.py tests/config_cnn_exp5_1979_2014.yaml > logs/cnn_exp5_1979_2014.log 2>&1 &
 
# echo "Launching 3. CNN Exp 3..."
# python3 -u src/predict.py tests/config_cnn_exp3_1979_2014.yaml > logs/cnn_exp3_1979_2014.log 2>&1 &

# echo "Launching 4. ViT Exp 2..."
# python3 -u src/predict.py config_vit_exp2_1979_2014.yaml > logs/vit_exp2_1979_2014.log 2>&1 &

# echo "Launching 5. ViT Exp 11 (Global Norm)..."
# python3 -u src/predict.py config_vit_exp11_1979_2014.yaml > logs/vit_exp11_1979_2014.log 2>&1 &

# echo "Launching 6. ViT Exp 14 (Deep Long)..."
# python3 -u src/predict.py config_vit_exp14_1979_2014.yaml > logs/vit_exp14_1979_2014.log 2>&1 &

# echo "Launching 7. ViT Exp 15 (High WD)..."
# python3 -u src/predict.py config_vit_exp15_1979_2014.yaml > logs/vit_exp15_1979_2014.log 2>&1 &

echo "All tasks launched in background. Waiting for complete completion..."
wait

echo "=================================================="
echo "All Inferences Completed Successfully in Parallel!"
echo "Results are stored in results/1979_2014/"
echo "=================================================="
