# Changelog

All notable changes to this project are documented in this file, which is updated after each commit.

---

## [3dc7154] - 2026-05-17
### Added
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
