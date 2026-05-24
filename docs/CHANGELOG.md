<!-- Author: M. El Aabaribaoune (@um6p) -->
# Changelog

All notable changes to this project are documented in this file, which is updated after each commit.

---

## [Scenario 3 & Loss Masking] - 2026-05-24
### Added
- **Macro-Region Loss Masking (Scenario 3)**: Implemented training models on the full domain while restricting loss calculations to merged sub-domain shapefiles. Built two macro-regions:
 - **Macro-region A (`north_northeast`)**: Merged North and Northeast shapefiles.
 - **Macro-region B (`east_south`)**: Merged East and South shapefiles.
- **Support for Multi-Shapefile Loss Masks**: Enhanced the mask building pipeline in `train.py` to accept lists of shapefiles, automatically load them, concatenate their geometries using `geopandas` and `pandas`, and project/rasterize the dissolved shape onto the target grid.
- **List-Based Shapefile Parser**: Updated `config.py` to parse list-based shapefile inputs and resolve relative paths correctly. Added checks to prevent redundant suffix appending if the experiment name already ends with the macro-region name.
- **SLURM Job Generator scripts**: Created `generate_lossmask_setups.py` and `generate_scenario3_setups.py` to automatically compile configurations and generate cluster submission scripts. Generated scripts use absolute paths to avoid SLURM spool folder failures.
- **Job Queuing**: Automatically generated and queued all 12 sub-domain loss-mask jobs and 6 Scenario 3 macro-region loss-mask jobs for `cnn_exp3`, `cnn_exp5`, and `vit_precip_exp21_best_hybrid`.

---

## [f4a9b2c] - 2026-05-17
### Added
- **Regional vs Unified Post-Processing Framework**: Created a dedicated comparison configuration [config_regional_vs_unified.yaml](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/tests/config_regional_vs_unified.yaml) to evaluate the spatial training impact by directly comparing unified Moroccan models against their sliced regional counterparts (glued back from separate North and South training sessions) over testing period (2006–2020).
- **Customized Seasonal and Extremes Suite**: Configured the pipeline to calculate the complete `mean` metrics suite (bias, rmse, correlation) and the exact subset of requested `extreme` indices (`cdd` and `r95_freq`) across all seasons (Annual, DJF, MAM, JJA, SON).
- **Comprehensive Splicing and GLM Splicing**: Updated [glue_regional_predictions.py](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/main/glue_regional_predictions.py) to support splicing of Generalized Linear Models (`glm_precip_l2`) alongside the deep learning CNN and ViT models, and ran the script to assemble regional predictions for all four models.
- **SLURM Automation**: Built and submitted [job_postproc_regional_vs_unified.sh](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/job_postproc_regional_vs_unified.sh) on the compute partition, currently running and compiling plots.

---

## [d3e5b4a] - 2026-05-17
### Changed
- **Redefined Sweep Plot Aesthetics**: Overhauled `postproc/utils.py` visualization pipeline (`plot_temporal_evolution`, `plot_metric_boxplot`, `plot_monthly_cycle`, `plot_intensity_distribution_log`, `plot_intensity_distribution_linear`) to support high-contrast dynamic colors (cycling over colormap `tab20`), variable line styles, and custom marker configurations for complex multi-model comparison sweeps.
- **Rotated Tick Labels and External Legend Layout**: Implemented automatic 45-degree rotation and shrunk font size (`fontsize=8`) for boxplot x-ticks, preventing severe overlaps when evaluating 12+ configurations. Moved all line plot legends to the right side of the plot using `bbox_to_anchor` layout to avoid data occlusion.
- **Enabled Sweep Mean Metrics**: Modified `main/generate_postproc.py` to enable the `mean` metrics (bias, rmse, correlation) suite alongside extreme metrics for the UNet sweeps, and resolved a bash wait-loop variable escaping syntax bug in generated SLURM scripts.
- **Resubmitted Post-Processing Jobs**: Resubmitted all 6 UNet sweep post-processing jobs to the compute partition, executing successfully in real-time.

---

## [7cda1b3] - 2026-05-17
### Added
- **Regional Prediction Gluing Tool**: Created [glue_regional_predictions.py](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/main/glue_regional_predictions.py) to automatically load and concatenate `North` ($28^\circ\text{N} \to 37^\circ\text{N}$) and `South` ($21^\circ\text{N} \to 28^\circ\text{N}$) NetCDF predictions along the latitude dimension for deep learning models (CNN Exp3, CNN Exp5, and ViT). This dynamically constructs a seamless, unified Moroccan-domain prediction NetCDF compatible with the main post-processing and metric plotting pipelines.
- **GLM Regional Split Redundancy Verification**: Formally verified that spatial regional splitting is redundant for Generalized Linear Models (`PixelWiseGLM`). Since the model fits each pixel $(i,j)$ completely independently without spatial weight sharing:
 $$\text{Model}_{(i, j)} = \text{GLM}(\text{Predictors}_{(i, j)}, \text{Precipitation}_{(i, j)})$$
 the fitted coefficients in regional runs are mathematically identical to those in the unified run.
### Changed
- **Regional Evaluation Strategy**: Excluded GLM regional runs from the spatial slicing comparison to focus purely on the deep learning models (CNN and ViT) where global spatial weights (convolutions and self-attention heads) are shared and heavily impacted by regional training domains.

---

## [2def42d] - 2026-05-17
### Added
- **Regional Split Evaluation Setup (North vs. South)**: Implemented a robust comparative framework to evaluate spatial training domain impacts on downscaling performance (comparing a single unified Morocco model against regionally specialized ones).
- **Auto-Generation Pipeline**: Created [generate_regional_setups.py](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/main/generate_regional_setups.py) to dynamically construct regional divided coordinates:
 - **North**: $28^\circ\text{N} \to 36^\circ/37^\circ\text{N}$ (covering mountainous Atlas and Mediterranean storms).
 - **South**: $21^\circ\text{N} \to 28^\circ\text{N}$ (covering hyper-arid Saharan provinces).
- **Regional Configurations**: Automatically generated divided YAML files under model-specific subdirectories:
 - `main/configs/cnn/regional/` (`cnn_exp3_north`, `cnn_exp3_south`, `cnn_exp5_north`, `cnn_exp5_south`)
 - `main/configs/vit/regional/` (`north`, `south`)
 - `main/configs/glm/regional/` (`north`, `south`)
- **Regional SLURM Scripts**: Automatically generated 8 optimized SLURM script entries under `main/scripts/regional/` for both standard GPU partitions and compute CPU partitions. All 8 regional training/validation jobs have been successfully submitted to the cluster queue.

---

## [3dc7154] - 2026-05-17
### Added
- **Production SLURM Submission Scripts**: Added 4 optimized SLURM job scripts inside `main/scripts/` to train and evaluate production models with the new land-mask loss configurations on the GPU and CPU partitions:
 - `job_gpu_cnn_exp3.sh`: Runs CNN Exp 3 on GPU.
 - `job_gpu_cnn_exp5.sh`: Runs CNN Exp 5 on GPU.
 - `job_gpu_vit.sh`: Runs ViT Best Hybrid on GPU.
 - `job_cpu_glm.sh`: Runs GLM L2 on CPU with optimized 32 parallel task cores.
- **Land-Sea Masking during Training & Validation**: Integrated spatial land-sea masking into loss calculations for CNN, ViT, and GLM models, optimizing the training purely for land-surface downscaling.
- **`regionmask` Integration**: Used Natural Earth 1:110m land polygons on target coordinate grids `lon_out` and `lat_out` inside [train.py](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/main/train.py) to automatically build the spatial mask tensor.
- **Masked Custom PyTorch Loss Functions**: Updated all major loss functions in [losses.py](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/main/src/core/losses.py) and [vit_arch.py](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/main/src/models/vit_arch.py) (including `BernoulliGammaLoss`, `GaussianLoss`, `AsymmetricMSELoss`, `IntensityWeightedMSELoss`, and `HurdleLoss`) to accept an optional `mask` parameter.
- **Broadcasting & Normalized Loss**: Implemented custom masking multiplication and normalization:
 $$\text{Loss} = \frac{\sum (\text{Loss}_{\text{element-wise}} \times \text{Mask})}{\text{Batch Size} \times \sum \text{Mask}}$$
 ignoring ocean cells and maintaining stable, unbiased land gradients.
- **GLM Optimization**: Enhanced [glm.py](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/main/src/models/glm.py) to skip parallel GLM pixel-wise training tasks for ocean points, reducing total tasks from 27,000 to **17,458 land pixels** and speeding up GLM fitting significantly.

---

## [7bd351e] - 2026-05-17
### Added
- **`CLEANING_STATE.md`**: Created a central documentation file containing a complete selection guide of the best configurations (GLM: `glm_precip_l2`, CNN: `cnn_exp3`/`cnn_exp5`, ViT: `vit_precip_exp21_best_hybrid`), the cleaned config directory structure, and details on safely executing directory cleanups.

---

## [a96bede] - 2026-05-16
### Changed
- **UNet Post-Processing Sweep Setup**: Created and submitted 6 automated SLURM jobs to run `postproc.py` for all 72 UNet hyperparameter sweep variants across the Annual period (2006-2020), calculating mean metrics.
- **Automatic Wait-Loop Dependency**: Added Bash wait-loops in SLURM scripts to dynamically monitor training progress and trigger evaluation as soon as training outputs (`.nc` files) become available.

---

## [d55a639] - 2026-05-09
### Added
- **Massive UNet Sweep & Architecture Fix**: Resolved spatial resolution issues in the UNet upsampling layers (aligning with CNN and ViT native high-resolution input patterns).
- **Sweep Generation Scripts**: Implemented a standalone Python script to generate all 72 configs and job scripts to comprehensively sweep batch sizes `[64, 128, 512, 1024]` and learning rates `[1e-3, 1e-4, 1e-5]` across 6 baseline UNet setups (`exp1` to `exp6`).

---

## [363c0d8] - 2026-05-09
### Changed
- **Repository Reorganization**: Standardized repository layout by clean archiving of test configs. Moved obsolete configs into model-specific `tests/` directories while maintaining relative file path compatibility.
