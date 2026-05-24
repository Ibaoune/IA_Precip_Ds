<!-- Author: M. El Aabaribaoune (@um6p) -->
# Climatological & Extreme Precipitation Downscaling Evaluation Pipeline

This directory contains the modular post-processing, evaluation, and visualization pipeline for the statistical downscaling of precipitation over Morocco. It enables standard climatological calculations and extreme precipitation indices assessment, comparing downscaled predictions (from GLM, CNN, U-Net, and ViT) to observations (MSWEP) and raw General Circulation Model (GCM) baselines.

---

## 📂 Directory Structure

* **`configs/`**: Configuration profiles defining evaluation scenarios, temporal/spatial limits, and active metrics.
  * [`config_retained.yaml`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/config_retained.yaml): Config for evaluating retained models (GLM, CNN, U-Net, ViT) against MSWEP on the 10km grid.
  * **MSWEP 10km Comparison**:
    * [`config_infer_lmdz250_to_mswep.yaml`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/config_infer_lmdz250_to_mswep.yaml): Downscaled LMDZ250 predictions + raw LMDZ250 vs. MSWEP observations. (Includes GLM, CNN, U-Net, ViT).
    * [`config_infer_lmdz35_to_mswep.yaml`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/config_infer_lmdz35_to_mswep.yaml): Downscaled LMDZ35 predictions + raw LMDZ3.5 vs. MSWEP observations. (Includes GLM, CNN, U-Net, and ViT).
  * **Direct GCM Parent 10km Comparison**:
    * [`config_infer_lmdz250_to_raw10km.yaml`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/config_infer_lmdz250_to_raw10km.yaml): Downscaled predictions vs. their parent GCM baseline (interpolated to 10km). (Includes GLM, CNN, U-Net, ViT).
    * [`config_infer_lmdz35_to_raw10km.yaml`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/config_infer_lmdz35_to_raw10km.yaml): Downscaled predictions vs. their parent GCM baseline (interpolated to 10km). (Includes GLM, CNN, U-Net, and ViT).
  * **Native Coarse Grid Comparison ("Rough")**:
    * [`config_infer_lmdz250_rough.yaml`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/config_infer_lmdz250_rough.yaml): Evaluates downscaled predictions (coarsened) against native GCM coarse grid (`16x9`). (Includes GLM, CNN, U-Net, ViT).
    * [`config_infer_lmdz35_rough.yaml`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/config_infer_lmdz35_rough.yaml): Evaluates downscaled predictions (coarsened) against native GCM coarse grid (`65x63`). (Includes GLM, CNN, U-Net, and coarsened ViT).
  * `tests/`: Lightweight testing configurations.
* **`datasets/`**: Local cache for aligned, unit-converted GCM baselines and coarsened predictions.
  * `raw_lmdz250_10km.nc` / `raw_lmdz35_10km.nc`: Bilinearly regridded GCM baselines at 10km.
  * `raw_lmdz250_coarse.nc` / `raw_lmdz35_coarse.nc`: Cleaned native coarse-grid GCM baselines (in daily units).
  * `coarse_lmdz250/` / `coarse_lmdz35/`: Downscaled predictions coarsened to the respective GCM grid using bilinear mapping.
* **`scripts/jobs/`**: Slurm shell execution files.
  * [`job_postproc_retained.sh`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/scripts/jobs/job_postproc_retained.sh): Master SLURM script for retained model evaluation.
  * [`job_postproc_infer_template.sh`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/scripts/jobs/job_postproc_infer_template.sh): Reusable SLURM template executing postprocessing for any specified config file.
  * `job_regional_summary.sh` / `job_seasonal_summary.sh`: Utility summary scripts.
* **`src/`**: Central python source code.
  * `postproc.py`: Master runner orchestrating calculation of all metrics (e.g. Bias, RMSE, extremes) across all active regions.
  * `utils.py`: Geographical masking, color scaling, name mappings, and central plotting functions.
  * `mean/`, `extreme/`, `boxplots/`, `climatology/`, `seasonal/`, `temporal/`: Individual metric calculation and sub-plotting directories.
* **`prepare_comparison_datasets.py`**: Automated pipeline for interpolating, unit-converting, and normalizing GCM datasets.
* **`shape_files/`**: Morocco shapefiles used for spatial geographical masking.
* **`results/`**: Outputs categorized as `results/<experiment_name>/<start_date>_<end_date>/<region_name>/`.

---

## 🛠️ Data Preparation & Alignment Pipeline

General Circulation Models (GCMs) write data on varied coarse grids and with inconsistent physical unit structures. The pipeline provides [`prepare_comparison_datasets.py`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/prepare_comparison_datasets.py) which handles this automatically:

1. **Daily Unit Conversion**: Coarse GCM outputs express precipitation in flux density units (`kg/(s*m2)`). The script multiplies by **`86400.0`** to convert them into daily precipitation depth (**`mm/day`**), matching MSWEP and downscaled outputs.
2. **Time Coordinate Normalization**: Raw LMDZ daily fields are stamped at noon (`12:00:00`), whereas downscaled results and MSWEP observations are daily averages stamped at midnight (`00:00:00`). The script floors all timestamps to daily midnights (`dt.floor('D')`) to prevent `xarray.align` from dropping dates.
3. **Double-Ended Spatial Mapping**:
   * **Bilinear Regridding**: Maps coarse GCMs up to the prediction 10km grid (`160x180`).
   * **Bilinear Coarsening**: Decimates downscaled outputs down to coarse resolutions (`16x9` for LMDZ250, `65x63` for LMDZ3.5) to allow native rough-grid evaluations. This includes coarsening the ViT prediction dataset (`vit_lmdz_250_present_true.nc`) directly onto the coarse LMDZ3.5 grid (`datasets/coarse_lmdz35/vit_lmdz35_coarse.nc`).

Run it using:
```bash
python3 prepare_comparison_datasets.py
```

---

## 🎨 Standardized Climatological Aesthetics

All plotting modules in `src/` utilize centralized styling defined inside [`src/utils.py`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/src/utils.py):

* **Scientific Name Translation (`get_display_name`)**: Converts model and database keys dynamically to standard scientific formatting inside titles, legends, and axes:
  * Downscaling models: `U-Net`, `ViT`, `GLM`, `CNN`.
  * Baseline baselines: `MSWEP`, `LMDZ 250`, `LMDZ 3.5`.
* **Persistent Model Colors (`GLOBAL_MODEL_COLORS`)**: Ensures an identical, high-contrast color palette follows each model across every spatial, boxplot, seasonal cycle, and daily distribution plot:
  * **`MSWEP`**: `#000000` (Black)
  * **`LMDZ 250`**: `#F4A261` (Warm Orange)
  * **`LMDZ 3.5`**: `#E76F51` (Warm Red-Orange)
  * **`GLM`**: `#2A9D8F` (Teal/Green)
  * **`CNN`**: `#457B9D` (Light Blue)
  * **`U-Net`**: `#1D3557` (Dark Blue)
  * **`ViT`**: `#E63946` (Bright Red)

---

## 🚀 Execution Guide

Ensure your conda environment is activated before running tasks:
```bash
conda activate clean_env_Pytorch
```

### Local Run
Submit any configuration directly from the command line:
```bash
python3 -u src/postproc.py configs/config_infer_lmdz250_to_mswep.yaml
```

### SLURM Cluster Submission

#### 1. Submit Retained Evaluation Job
```bash
sbatch scripts/jobs/job_postproc_retained.sh
```

#### 2. Submit Inference & GCM Comparison Suite (Parallel Run)
You can launch any of the GCM comparison jobs using the template runner script:
```bash
# Compare Downscaled + Raw LMDZ250 to MSWEP
sbatch scripts/jobs/job_postproc_infer_template.sh configs/config_infer_lmdz250_to_mswep.yaml

# Compare Downscaled + Raw LMDZ35 to MSWEP
sbatch scripts/jobs/job_postproc_infer_template.sh configs/config_infer_lmdz35_to_mswep.yaml

# Compare Downscaled LMDZ250 models directly against parent raw model at 10km
sbatch scripts/jobs/job_postproc_infer_template.sh configs/config_infer_lmdz250_to_raw10km.yaml

# Compare Downscaled LMDZ35 models directly against parent raw model at 10km
sbatch scripts/jobs/job_postproc_infer_template.sh configs/config_infer_lmdz35_to_raw10km.yaml

# Evaluate models against LMDZ250 natively on its coarse 16x9 grid
sbatch scripts/jobs/job_postproc_infer_template.sh configs/config_infer_lmdz250_rough.yaml

# Evaluate models against LMDZ35 natively on its coarse 65x63 grid
sbatch scripts/jobs/job_postproc_infer_template.sh configs/config_infer_lmdz35_rough.yaml
```
