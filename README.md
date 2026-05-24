<!-- Author: M. El Aabaribaoune (@um6p) -->
# Precipitation Downscaling Project

This repository hosts a comprehensive, end-to-end framework for statistical and deep learning-based climate downscaling over Morocco. It is designed to take coarse-resolution Global Circulation Model (GCM) outputs (like LMDZ) or reanalysis data (ERA5) and downscale them into high-resolution precipitation fields (matching the MSWEP reference dataset).

The project is structured into distinct, modular components, each handling a specific phase of the climate modeling pipeline.

## 📁 Repository Structure

The project is divided into the following main directories. **Click on any directory link below to read its specific, detailed documentation.**

### 1. [`main/`](main/README.md) - Training & Evaluation Engine
The core workspace for defining, training, and evaluating deep learning and statistical models.
- **Supported Models**: Convolutional Neural Networks (CNN), U-Net, Vision Transformers (ViT), and Generalized Linear Models (GLM).
- **Features**: Highly modular PyTorch-based training loops, robust config-driven execution, and automated batch test runners (`master_test_runner.py`).

### 2. [`inference/`](inference/README.md) - Unified Inference Engine
A robust deployment engine to apply trained models onto historical datasets or future GCM projections.
- **Features**: Automatic spatial alignment, scaling delta mapping (SDM) for GCM bias correction, and parallel SLURM execution across multiple scenarios (e.g., `era5_present`, `lmdz_35_present`, `lmdz_250_present`).

### 3. [`postproc/`](postproc/README.md) - Climate Metrics & Post-Processing
A comprehensive evaluation suite to calculate climatological metrics and visualize model performance.
- **Metrics**: Computes bias, RMSE, correlation, and extreme indices (CDD, R95, R99, etc.).
- **Visualizations**: Automatically generates spatial maps, temporal evolution line charts, and boxplots across dynamically defined sub-regions of Morocco.

### 4. [`datasets/`](datasets/README.md) - Data Loading & Normalization
*(Under Development)* Dedicated modules and scripts handling raw dataset ingestion, strict normalization schemes, and chunking logic.

### 5. [`lit_version/`](lit_version/README.md) - PyTorch Lightning Legacy
An alternative/legacy version of the training framework implemented using PyTorch Lightning. Useful for those who prefer the Lightning abstraction layer for multi-GPU scaling.

---

## 📚 General Documentation

For general guides, methodology descriptions, and repository maintenance:
*   **[Loss Masking Guide](docs/LOSS_MASKING_GUIDE.md)**: Detailed technical guide on regional loss masking, single sub-regions, and merged macro-regions (Scenario 3).
*   **[Regional Splits & Splicing Guide](docs/REGIONAL_EVALUATION_GUIDE.md)**: Guide on cropped regional splitting ($28.0^\circ\text{N}$ boundary), prediction gluing (`glue_regional_predictions.py`), and GLM spatial redundancy.
*   **[Cleaning State & Config Guide](docs/CLEANING_STATE.md)**: Selection guide identifying top models, config directories organization, and safe cleanup instructions.
*   **[Project Changelog](docs/CHANGELOG.md)**: Detailed historical log of features added and changed in the repository.

---

## 🚀 Getting Started

### Environment Setup
Make sure your Python environment is activated. The code relies heavily on `torch`, `xarray`, `netCDF4`, `geopandas`, and `cartopy`.
```bash
conda activate clean_env_Pytorch
```

### Workflow Overview
1. **Train a Model**: Navigate to `main/`, configure your YAML file, and run `python train.py configs/<model>/config.yaml`.
2. **Run Inference**: Navigate to `inference/`, configure your target datasets in `config.yaml`, and run `sbatch run_inference.sh`.
3. **Analyze Results**: Navigate to `postproc/`, configure `config.yaml` to point to your new inference outputs, and run `sbatch job_postproc.sh` to generate metric plots.

---
*Developed for climate modeling research over Morocco.*
