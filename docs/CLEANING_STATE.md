# Cleaning State & Config Selection Guide

This document summarizes the current status of all downscaling experiments, identifying the best-performing models, their corresponding configuration files, and instructions for safely cleaning up directories without losing valuable work.

---

## 1. Summary of Model Configurations

| Model | Chosen Main Config File | Experiment Name | Key Features / Hyperparameters | Status / Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **GLM** | `main/configs/glm/config.yaml` | `glm_precip_l2` | Bernoulli-Gamma loss, Gridbox normalization, L2 weight decay regularization (`value: 0.01`). | **Best GLM baseline.** Outperformed simple L1 Lasso regularizations. |
| **CNN** | `main/configs/cnn/cnn_exp3.yaml` | `cnn_exp3` | Bernoulli-Gamma, Gridbox norm, Cosine LR scheduler, Group Norm, Dropout `0.3`, `batch_size: 512`, `lr: 1e-3`. | **Top CNN Candidate.** Strong, robust learning curves. |
| | `main/configs/cnn/cnn_exp5.yaml` | `cnn_exp5` | Bernoulli-Gamma, Gridbox norm, Cosine LR scheduler, Group Norm, Dropout `0.2`, `batch_size: 64`, `lr: 1e-4`. | **Top CNN Candidate.** Excellent fine-tuned performance. |
| **ViT** | `main/configs/vit/config.yaml` | `vit_precip_exp21_best_hybrid` | Hybrid extraction, Global norm, Embedding size `64`, `8` layers, Group Norm, Dropout `0.2`, `weight_decay: 1e-3`. | **Best Overall Model.** Superior spatial preservation and physical downscaling stability. |
| **UNet** | *Under Sweep (Pending)* | `unet_exp1` to `unet_exp6` | Reduction to 3 MaxPool layers (fixes spatial dimension crash), training sweep across Batch Size `[64, 128, 512, 1024]` and LR `[1e-3, 1e-4, 1e-5]`. | **Hyperparameter Sweep Running.** Evaluating 72 combinations. |

---

## 2. Config Files Reorganization Directory Structure

To clean up the directories and keep only the production-ready setups, configurations have been organized as follows:

### `main/configs/`
- **`glm/`**
  - `config.yaml` ➔ **Main production config** (based on `alpha_l2`).
  - `tests/` ➔ Contains all older exploration configs (`alpha_l1.yaml`, `alpha_l2.yaml`, `interp_bilinear.yaml`, `interp_nearest.yaml`, `config_old.yaml`, `test.yaml`). *Can be archived or cleaned.*
- **`cnn/`**
  - `cnn_exp3.yaml` ➔ **Main config for Exp 3**.
  - `cnn_exp5.yaml` ➔ **Main config for Exp 5**.
  - `regional/` ➔ Bounding box cropped regional configurations.
  - `loss_mask/` ➔ Full-domain configurations with sub-domain loss masking.
  - `scenario3/` ➔ Full-domain configurations with merged macro-region loss masking (Scenario 3).
  - `tests/` ➔ Contains hyperparameter exploration configs (`gridbox_LR_scheduler.yaml`, `gridbox_dropout.yaml`, etc., and `config_old.yaml`, `test.yaml`). *Can be archived or cleaned.*
- **`vit/`**
  - `config.yaml` ➔ **Main production config** (based on `test_exp21_best_hybrid`).
  - `regional/` ➔ Bounding box cropped regional configurations.
  - `loss_mask/` ➔ Full-domain configurations with sub-domain loss masking.
  - `scenario3/` ➔ Full-domain configurations with merged macro-region loss masking (Scenario 3).
  - `tests/` ➔ Contains experiments 1 to 25 (`test_exp1.yaml` to `test_exp25_hybrid_fast_cosine.yaml`, etc., and `config_old.yaml`, `test.yaml`). *Can be archived or cleaned.*
- **`unet/`**
  - Keep as is during sweep evaluation. `tests/sweep/` holds all 72 active configurations.

---

## 3. Post-Processing Configuration State (`postproc/`)

### Current State:
- `postproc/tests/config_vit_hybrids_vs_cnn.yaml` has been cleaned to evaluate **only** the top-performing runs:
  1. `unet` (currently utilizing `cnn_exp3` as placeholder).
  2. `cnn` (utilizing `cnn_exp5`).
  3. `glm` (utilizing `glm_precip_l2`).
  4. `ViT_Exp21_Best` (utilizing `vit_precip_exp21_best_hybrid`).
- Other test configurations (like `vit_precip_exp22_hybrid_deep_reg`) have been removed from the comparison.

- **Active Postproc Sweep Jobs:** 6 lightweight jobs (`pp_unet_1` to `pp_unet_6` under job IDs `7040819` to `7040824`) are running `postproc.py` on the Annual mean metrics for all 72 UNet variants.

---

## 4. Directory Cleanup Recommendations

If you plan to clean up the entire `papers/downscaling` workspace, here is what you can safely do:

1. **Keep `main/src/` & `main/configs/`**: These contain your core architectures, models, and selected configs.
2. **Keep the Chosen Weights**: In `main/results/` (or `inference/models/`), ensure you **do not delete**:
   - `vit_precip_exp21_best_hybrid` weights
   - `cnn_exp3` and `cnn_exp5` weights
   - `glm_precip_l2` weights
   - Keep the active `unet_expX` training weights currently in `main/results/`.
3. **Safely Delete**:
   - Old tensorboard logs or debug figure outputs from obsolete experiments (e.g., intermediate `vit_precip_exp` 1 to 20 folders if no longer needed for paper writing).
   - Old text logs under model-specific `logs/` directories that aren't tied to active runs.
