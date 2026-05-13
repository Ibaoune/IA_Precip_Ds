# Inference Pipeline for Deep Learning Downscaling

This directory contains the complete inference pipeline for deploying your trained deep learning downscaling models (UNet, CNN, GLM, ViT). It takes trained weights (learned by mapping ERA5 to MSWEP) and applies them to new periods or entirely new predictor datasets like Global Climate Models (GCMs, e.g., LMDZ).

## Is this the best way to test the weights?

**Yes.** Doing inference in the model space (the gridded domain) using weights trained on ERA5-to-MSWEP is the scientifically standard and most robust approach to climate downscaling. 

Here is why this pipeline is correctly designed for that purpose:
1. **Standardization:** The pipeline caches the mean and standard deviation of the *training* ERA5 data. When new data is fed in, it is standardized using these historical ERA5 statistics. This ensures the neural network receives data in the exact same distribution it was trained on.
2. **Bias Correction:** When switching predictors (e.g., feeding LMDZ data into a model trained on ERA5), GCMs inherently have systemic biases. This pipeline includes an elegant Bias Correction module (`bias_correction.py` using Scaling Delta Mapping - SDM). It adjusts the GCM data to match the ERA5 climatology *before* it goes into the neural network, preventing out-of-distribution errors.

## Pipeline Architecture

The main entry point is `predict.py`, which is driven by `config.yaml` and executes the following workflow for each defined "scenario":

1. **Load Data (`data_loading.py`)**: Loads the new predictor dataset (ERA5 test set or LMDZ projections).
2. **Interpolation (`interpolation.py`)**: Interpolates the inputs to match the target spatial resolution if necessary.
3. **Bias Correction (`bias_correction.py`)**: (If enabled) Applies SDM using historical GCM and historical ERA5 data to correct systemic biases in the predictors.
4. **Standardization (`bias_correction.py`)**: Standardizes the predictors using the cached historical ERA5 statistics.
5. **Model Loading (`utils.py` & `models/`)**: Loads the architecture and the trained weights (`.pth` or `.pkl`).
6. **Prediction (`predict.py`)**: Batches the standardized inputs through the neural network. If using the `bernoulli_gamma` loss, it calculates the final precipitation from the occurrence, shape, and scale parameters.
7. **Export (`predict.py`)**: Saves the downscaled predictions as a `.nc` file in the `results/output` directory.

## How to Run

### 1. Update the Configuration
Open `config.yaml` and update the paths to match your current environment. Pay special attention to:
*   `paths.root_dir`
*   `paths.shapefile_path`
*   `paths.results_dir` (Where your trained models are saved)
*   `prediction.scenarios` (Define the periods and datasets you want to run inference on. Update the `folder` paths to your actual LMDZ/ERA5 locations).

### 2. Launch the Job
Once configured, simply submit the inference job to SLURM:

```bash
sbatch predict.sh
```

The script will iterate through all scenarios defined in your `config.yaml` and generate a `[scenario_name].nc` file for each.
