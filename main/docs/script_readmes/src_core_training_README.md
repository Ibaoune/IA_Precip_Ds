# Documentation for `src/core/training.py`

## Overview
==========================================================
 Script: training.py
 Author: M. El Aabaribaoune (@um6p)
 Description:
     Defines the training loop, loss computation, and
     early stopping for the downscaling model.

 Design:
     - Model-agnostic (ViT, CNN, UNet, etc.)
     - GPU / CPU compatible
==========================================================

## Functions
### `def _build_model(...)`
Factory function for model instantiation.

### `def train_model(...)`
No description.

## How to Modify
If you need to make changes to `training.py`:
- **Logic changes**: Locate the corresponding function. Update its docstring if you change its signature or behavior.
- Ensure any related imports in other files are updated if you change function/class names.