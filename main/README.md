<!-- Author: M. El Aabaribaoune (@um6p) -->
# Precipitation Downscaling: Main Training & Evaluation Environment

This directory (`main/`) is the central workspace for training, evaluating, and testing deep learning and statistical models for precipitation downscaling. It handles the end-to-end pipeline: from loading and preprocessing coarse-resolution climate data (e.g., ERA5, GCM) to predicting and evaluating high-resolution precipitation fields (e.g., MSWEP-like grids).

## Directory Structure

```text
main/
├── train.py # Main script for training models
├── eval.py # Main script for model evaluation/inference
├── master_test_runner.py # Automated test runner across multiple configs
├── configs/ # YAML configuration files (organized by model)
│ ├── unet/
│ ├── cnn/
│ ├── vit/
│ └── glm/
├── src/ # Core source code modules
│ ├── core/ # Training, evaluation, losses, and utils
│ ├── data/ # Data loading, preprocessing, and interpolation
│ └── models/ # Neural Network & Statistical architectures (CNN, U-Net, ViT, GLM)
├── scripts/ # Bash/SLURM scripts for cluster execution
├── docs/
│ └── script_readmes/ # Detailed documentation for every individual .py script
└── results/ # Auto-generated directory for outputs, models, and logs
```

## Script-Level Documentation (Where to Modify Code)

If you need to dive into the code and modify specific behaviors, we have dedicated READMEs for every Python file in the repository. These docs will tell you exactly what classes and functions are available, and where to make changes.

**[View Script Documentation Index](docs/script_readmes/INDEX.md)**

## Supported Models

The `src/models/` directory implements multiple architectures:
- **U-Net** (`unet_arch.py`): Primary architecture for spatial precipitation prediction.
- **Vision Transformer (ViT)** (`vit_arch.py`, `vit_arch_or.py`): Transformer-based architectures for capturing global spatial dependencies.
- **CNN** (`cnn.py`): Convolutional Neural Network baseline.
- **GLM** (`glm.py`): Generalized Linear Model (statistical baseline).

## Configuration System

All runs are controlled via YAML configuration files located in `configs/`. Each model has its own sub-directory containing standard `config.yaml` and test-specific files (in the `tests/` sub-directories).

Key parameters managed in the configs:
- **Experiment settings**: Paths, variables (`precip`), scenarios (ERA5, HIST, SSPs).
- **Hyperparameters**: Learning rate, batch size, epochs, normalization mode, loss type, and schedulers.
- **Geospatial & Temporal**: Bounding box coordinates (lat/lon) and date ranges for training/testing.

### Regional & Macro-Region Loss Masking (Scenario 3)

The framework supports training models on the **full domain** (preserving complete spatial and boundary conditions) while calculating the loss and gradients exclusively on specific sub-domains. This prevents boundary edge artifacts in regional predictions.

To enable regional loss masking in a YAML configuration, add a `loss_mask` block under `training`:

```yaml
training:
 loss_mask:
 enable: true
 region: north_northeast # Name suffix appended to experiment output directory
 shapefile: # Single shapefile path (string) or a list of shapefiles to merge
 - postproc/shape_files/north.shp
 - postproc/shape_files/north_east.shp
```

When enabled, the training loop:
1. Loads the shapefile(s) and dissolves them into a single macro-region boundary.
2. Projects the geometry to match the target grid resolution and boundaries.
3. Combines it with the land/sea mask so that backpropagation only runs on land pixels within the selected region.
4. Saves all model checkpoints and validation outputs separately under `results/<experiment>_lossmask_<region>/` (unless the experiment name already ends with the region name).

Three script generators are available in `main/` to automate configuring regional and loss-mask jobs:
* **`generate_regional_setups.py`**: Generates cropped-region setups (physical bounding box cropped).
* **`generate_lossmask_setups.py`**: Generates full-domain setups with loss masking for standard sub-regions (`north`, `south`, `east`, `north_east`).
* **`generate_scenario3_setups.py`**: Generates full-domain setups with merged macro-region loss masking (`north_northeast` and `east_south`).

## Usage

### 1. Training a Model
Use `train.py` to train a model. By default, it looks for `config.yaml`, but you can pass any config path as an argument.

```bash
# Example: Train a U-Net model
python train.py configs/unet/config.yaml

# Example: Train a ViT model
python train.py configs/vit/config.yaml
```

*During training, the script will: load datasets, preprocess inputs, train the model using the appropriate device (CPU/GPU), and save checkpoints to `results/`.*

### 2. Evaluating a Model
Use `eval.py` to perform inference and compute evaluation metrics.

```bash
# Example: Evaluate a model using its test configuration
python eval.py configs/unet/test.yaml
```

*The evaluation step outputs test metrics and generates visualizations/data files in the corresponding `results/` directory.*

### 3. Automated Batch Testing
For running multiple hyperparameter combinations or evaluating several models sequentially, use `master_test_runner.py`.

```bash
python master_test_runner.py
```
This script dynamically modifies the `.sh` scripts in `scripts/`, launches batch jobs, and outputs a clean table detailing the status and save paths of each executed test.

### 4. Running on a SLURM Cluster
If you are running experiments on a computing cluster, utilize the pre-configured scripts in `scripts/`:

```bash
# Submit a CPU job
sbatch scripts/job_cpu.sh

# Submit a GPU job
sbatch scripts/job_gpu.sh
```

## Outputs and Results

All outputs are structured under `results/<experiment_name>/...` based on the parameters set in your config file. A typical result directory contains:
- **`models/`**: Saved model checkpoints (`.pth`).
- **`config.txt`**: A snapshot of the exact configuration used.
- **`output_data/`**: Evaluation metrics, raw predictions, and NetCDF files.
- **`output_figs/`**: Generated plots and visualizations.

## Extending the Pipeline

- **Adding a new model:** Add your architecture in `src/models/`, register it if needed, and create a new directory in `configs/` with your custom `config.yaml`.
- **Modifying data logic:** Adjust `src/data/data_loading.py` or `src/data/preprocessing.py` to accommodate new inputs or normalization techniques.
- **Adding new loss functions:** Update `src/core/losses.py` (e.g., tweaking the Bernoulli-Gamma loss).

---
*Note: Make sure your Python environment is activated (e.g., `conda activate clean_env_Pytorch`) and dependencies like PyTorch, xarray, and netCDF4 are installed before running scripts.*
