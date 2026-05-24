#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)
#SBATCH --job-name=ViT_hybrids_LMDZ250
#SBATCH --output=logs/out_hybrids_lmdz250_%j.log
#SBATCH --error=logs/out_hybrids_lmdz250_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

echo "=================================================="
echo "Starting Multi-Model Inference for Hybrids (LMDZ250 ONLY)"
echo "=================================================="

echo "Launching 1. ViT Exp 21 (Best Hybrid)..."
python3 -u predict.py config_vit_precip_exp21_best_hybrid_1979_2014.yaml > logs/vit_exp21_best_hybrid_lmdz250.log 2>&1 &

echo "Launching 2. ViT Exp 21 (Hybrid Base)..."
python3 -u predict.py config_vit_precip_exp21_hybrid_base_1979_2014.yaml > logs/vit_exp21_hybrid_base_lmdz250.log 2>&1 &

echo "Launching 3. ViT Exp 22 (Hybrid Deep Reg)..."
python3 -u predict.py config_vit_precip_exp22_hybrid_deep_reg_1979_2014.yaml > logs/vit_exp22_hybrid_deep_reg_lmdz250.log 2>&1 &

echo "Launching 4. ViT Exp 23 (Hybrid Bilinear Channel)..."
python3 -u predict.py config_vit_precip_exp23_hybrid_bilinear_channel_1979_2014.yaml > logs/vit_exp23_hybrid_bilinear_channel_lmdz250.log 2>&1 &

echo "All 4 tasks launched in background. Waiting for completion..."
wait

echo "=================================================="
echo "All Inferences Completed Successfully in Parallel!"
echo "=================================================="
