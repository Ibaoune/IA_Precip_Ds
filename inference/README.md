<!-- Author: M. El Aabaribaoune (@um6p) -->
# Unified Inference Engine (Downscaling)

This directory contains the universal and robust inference engine used to deploy all trained downscaling models (ViT, CNN, UNet, GLM).

The pipeline takes the weights trained on the historical task (ERA5 to MSWEP) and applies them to new periods or to climate projections from General Circulation Models (GCMs, e.g., LMDZ at different resolutions).

---

## Key Features & Robustness

1. **Automatic Architecture Loading (`train_config_path`):**
   Simply provide the path to the `config.txt` file from the model's training directory. The `predict.py` script automatically imports the correct hyperparameters (model type, loss, grid dimensions, interpolation type).

2. **Automatic Spatial Alignment:**
   The engine reads spatial boundaries (`lon_min/max`, `lat_min/max`) directly from the model's training configuration. This guarantees a perfect grid match and avoids NetCDF dimensional conflicts.

3. **Advanced Bias Correction (SDM):**
   When predicting on GCM data (LMDZ r35 or r250), the module applies the *Scaling Delta Mapping* method to correct systematic biases in GCM predictors against historical ERA5 climatology prior to inference.

4. **Granular Scenario Control (`enable: true/false`):**
   You can easily enable or disable individual datasets (ERA5, LMDZ r35, LMDZ r250) using a simple flag in the configuration.

---

## Directory Architecture

To maintain a clean and modular environment, the code has been reorganized as follows:

* **`src/`**: Contains the core Python inference logic.
  * `predict.py`: Main execution script.
  * `bias_correction.py`: Implements SDM for GCM predictors.
  * `data_loading.py`, `interpolation.py`, `preprocessing.py`, `regrid.py`, `utils.py`: Core utility modules for data manipulation.
  * `models/`: Subdirectory containing model architecture definitions.
* **`scripts/`**: Contains job submission scripts and auxiliary utilities.
  * `run_inference.sh`, `run_all_1979_2014.sh`: SLURM scripts to execute inference jobs.
  * `utils/`: Contains helper scripts like `generate_configs.py` and `update_config.py`.
* **`logs/`**: Stores all SLURM log files (`out_*.log`) generated during job execution.
* **`results/`**: The output directory where the final generated `.nc` prediction files are saved.
* **`tests/`**: Contains legacy configurations and specialized inference test scripts.

---

## Launching an Inference Job

To run an inference pipeline using your configuration (e.g., `config.yaml`), submit the unified bash script:

```bash
sbatch scripts/run_inference.sh
```

Execution logs will be automatically saved in the `logs/` directory, and output NetCDF files will be routed to your configured `output_dir` (usually under `results/`).
