# Reproducibility Guide: Statistical Downscaling of Climate Data

This document provides a complete, step-by-step guide to reproducing the results of the paper, from training the global baseline models to regional specialization, and finally to the LMDZ250 transferability tests (perfect-prognosis).

## Concept of Total Isolation
To guarantee absolute traceability for the journal submission, **this reproducibility pipeline is 100% isolated**. 
- It reads baseline configurations from `main/configs/reproduce/base/`.
- It saves all generated configurations, SLURM scripts, and trained model weights into dedicated `reproduce/` folders (e.g., `main/results/reproduce/`).
- This guarantees that reproducing the paper will never overwrite or mix with your exploratory development tests.

## Prerequisites
- **Environment:** Conda environment `clean_env_Pytorch` with PyTorch, Xarray, and dependencies.
- **Compute:** Access to a SLURM cluster with GPU partitions.
- **Data (Lustre):** 
  - ERA5 Predictors (`1979-2020_levels.nc`)
  - MSWEP Predictand (`mswep_1979_2020.nc`)
  - LMDZ250 GCM data for inference
  - Morocco Shapefiles (`data/shape_files/`)

---

## 🛠 The 5-Step Pipeline

We have developed a suite of numbered scripts located in the `scripts/reproduce/` folders. **Execute them in numerical order.**

### Phase 1: Training the Models

**Step 1: Global Training**
This script submits the training of the 4 retained baseline architectures (CNN, U-Net, ViT, GLM) over the entire Moroccan domain.
```bash
./main/scripts/reproduce/01_train_global.sh
```
*Outputs are saved in: `main/results/reproduce/global/`*

**Step 2: Regional Specialization (Loss Masking)**
This script generates the regional configurations automatically (using the shapefiles) and submits the trainings for the 4 models across the 4 climatic sub-regions.
```bash
./main/scripts/reproduce/02_train_regional.sh
```
*Outputs are saved in: `main/results/reproduce/regional/`*

*(Wait for all Phase 1 jobs to finish before proceeding to Phase 2, as the subsequent steps require the weights to exist.)*

---

### Phase 2: Evaluation on Historical Data (ERA5)

**Step 3: Post-processing & Heatmaps Generation**
This script creates the unified configuration to compare Global vs. Regional models using the weights from Steps 1 & 2, and launches the evaluation metrics computation (RMSE, Bias, Correlation, R95, CDD).
```bash
./postproc/scripts/reproduce/03_evaluate_era5.sh
```
*Metrics are saved in: `postproc/results/reproduce/era5/`*

*Note: Once the SLURM job `job_evaluate_era5.sh` is complete, you must manually run the summary plotting script to generate the final PDF heatmaps (Figures 5 & 6).*

---

### Phase 3: Transferability & Inference (LMDZ250)

**Step 4: Inference on GCM Predictors**
This step applies the Scaling Delta Mapping (SDM) bias correction on LMDZ250 predictors and infers precipitation using the reproducible weights.
```bash
./inference/scripts/reproduce/04_run_lmdz_inference.sh
```
*Predictions are saved as `.nc` files in: `inference/results/reproduce/lmdz/`*

**Step 5: Final Transferability Evaluation**
This script evaluates the performance of our models when forced by LMDZ250 predictors over the 1979-2014 period.
```bash
./postproc/scripts/reproduce/05_evaluate_lmdz.sh
```
*Metrics are saved in: `postproc/results/reproduce/lmdz/`*

---

## ⚡ Fast-Track: Testing Evaluation Scripts Immediately

Since training (Steps 01 & 02) can take 24 hours, you can use the **migration script** to copy your already-trained weights from your exploratory folders into the pristine `reproduce/` architecture. This allows you to immediately run Steps 03, 04, and 05.

```bash
./main/scripts/reproduce/00_migrate_existing_weights.sh
```
*This will safely `rsync` your pre-trained models into `main/results/reproduce/global/` and `regional/`.*
