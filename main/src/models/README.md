<!-- Author: M. El Aabaribaoune (@um6p) -->
# Models Directory — `main/src/models/`

This directory contains all neural network architecture definitions used in the
downscaling project. It is organized into three layers: production-ready files
for the models retained in the paper, shared infrastructure used across all
architectures, and archival files preserving the full experimental history.

---

## Files Overview

### Production — Retained architectures for the paper

**`unet_retained.py`**
Contains `UNet_Exp32_Parallel`, the officially retained U-Net architecture.
This is the only file that should be imported when running inference or
re-training the retained U-Net model. It is self-contained and depends only on
`AddCoords` from `unet_arch.py`. A full documentation header is included at the
top of the file, covering design rationale, training configuration, weight
paths, and the key lessons learned during debugging (see also
`docs/unet_troubleshooting_report.md`).

**`vit_arch.py`**
Contains `DownscalingViT`, the officially retained Vision Transformer
architecture (experiment `vit_precip_exp21_best_hybrid`). This file also
defines the supporting building blocks used exclusively by the ViT:
`BernoulliGammaLoss`, `PatchEmbedding`, `TransformerBlock`, `TransformerEncoder`,
and `UpsamplingDecoder`. Unlike the U-Net, the retained ViT was always in its
dedicated file and did not require a separate `vit_retained.py`.

**`cnn.py`**
Contains `CNN_Exp5`, the retained CNN baseline against which all U-Net and ViT
architectures are benchmarked. Implements a shallow convolutional stack
followed by a Dense layer.

**`glm.py`**
Contains the Generalized Linear Model baseline used as the lower-bound
reference in the evaluation.

---

### Infrastructure — Shared building blocks (do not remove)

**`unet_arch.py`**
Core infrastructure file imported by multiple modules across the codebase.
Contains the following reusable components:

- `AddCoords` — injects normalized (lat, lon) spatial coordinate channels into
  the input tensor to break translation invariance (CoordConv).
- `ConvBlock` — standard 3x3 convolutional block with BatchNorm/GroupNorm,
  ReLU, and optional Dropout.
- `UpSampleBlock` — bilinear upsampling followed by a 3x3 convolution to avoid
  checkerboard artefacts.
- `AttentionBlock` — attention gate for filtering skip connections.
- `UNet_V2` — a clean, fixed-kernel U-Net with bilinear upsampling.
- `UNet_CoordConv` — `UNet_V2` augmented with `AddCoords`.
- `Attention_UNet` — `UNet_V2` with attention gates on skip connections.
- `Doury_UNet` — deeper ELU-activation U-Net inspired by Doury et al. (2024).
- `UNet` — legacy architecture kept for backward compatibility.

Imported by: `unet_retained.py`, `unet_experiments.py`, `vit_experiments.py`,
`core/training.py`, `core/evaluation.py`, `inference/src/utils.py`.

---

### Archive — Full experimental history (reference only, not for production)

**`unet_experiments.py`**
Archive containing all 36 U-Net experimental configurations (Config1 through
Config36) developed and tested during the research phase. These range from
simple U-Net wrappers to complex hybrid architectures combining CNN branches
with U-Net decoders. This file is essential for reproducibility of the paper's
ablation study but should not be imported in production inference pipelines.

**`vit_experiments.py`**
Archive containing ViT experimental configurations (Exp1 through Exp8)
developed before settling on the `DownscalingViT` architecture. Kept for
reproducibility.

**`vit_arch_or.py`**
Original ViT implementation written before the codebase was refactored.
Preserved for backward compatibility with early experiment checkpoints.

---

## Retained Models Summary

### U-Net — `UNet_Exp32_Parallel`

- **Class**: `UNet_Exp32_Parallel` in `unet_retained.py`
- **Design**: Two parallel branches share a CoordConv-augmented input.
  Branch 1 (CNN_Exp5 replica) captures the global mean precipitation signal.
  Branch 2 (shallow 2-level U-Net encoder-decoder) captures local extremes and
  orographic residuals. The output is the element-wise sum of both branches.
- **Loss**: Bernoulli-Gamma (occurrence, log-shape, log-scale)
- **Normalization**: gridbox (pixel-wise ERA5 standardization)
- **Training config**: `main/configs/unet/retained/config.yaml`
- **Weights**: `main/results/unet/tests/tests_advanced_arch/unet_exp32_parallel/
  region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/
  test_2006_01_01_2020_12_31/
  gridbox_0.0001_bernoulli_gamma_200ep_wd_gc_dropout_cosine_gn/
  models/unet_exp32_precip.pth`
- **LMDZ inference output**: `inference/results/1979_2014/unet_exp32_parallel/
  unet_exp32_lmdz_250_present_true.nc`

### ViT — `DownscalingViT` (experiment `vit_precip_exp21_best_hybrid`)

- **Class**: `DownscalingViT` in `vit_arch.py`
- **Hyperparameters**: `emb_size=64`, `patch_size=4`, `num_layers=8`,
  `num_heads=4`, `global` normalization, Bernoulli-Gamma loss
- **Training config**: `main/configs/vit/retained/config.yaml`
- **Weights**: `main/results/vit/retained/vit_precip_exp21_best_hybrid/
  region_lat_21.0_37.0_lon_-18.0_0.0/train_1979_01_01_2005_12_31/
  test_2006_01_01_2020_12_31/
  global_0.001_bernoulli_gamma_25ep_lr_sched_wd_gc_dropout_cosine_gn/
  models/vit_precip.pth`

### CNN — `CNN_Exp5`

- **Class**: `CNN_Exp5` in `cnn.py`
- **Training config**: `main/configs/cnn/retained/config.yaml`

---

## Important Notes

Do not import from `unet_experiments.py` or `vit_experiments.py` in production
or inference code. These files exist solely for research reproducibility.

The `unet_arch.py` file is a shared dependency and must not be removed. Doing
so would break training, evaluation, and inference pipelines for all U-Net and
ViT architectures.
