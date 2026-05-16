# Datasets Directory

This directory serves as the centralized staging area for processed datasets and normalization statistics used by the downscaling models (ViT, CNN, U-Net, GLM).

Currently, it contains subdirectories for the models (`vit`, `cnn`, `unet`, `glm`), which are meant to hold specific metadata, scaling parameters, or minimal datasets required for reproducing runs across different machines without relying exclusively on the cluster's main Lustre storage.

## Contents
- **`vit/`**: Data assets specific to Vision Transformer experiments.
- **`cnn/`**: Data assets specific to Convolutional Neural Network experiments.
- **`unet/`**: Data assets specific to U-Net experiments.
- **`glm/`**: Data assets specific to Generalized Linear Model statistical baselines.

*(Note: The heavy NetCDF files themselves remain on the Lustre partition and should not be tracked by version control to avoid repository bloat).*
