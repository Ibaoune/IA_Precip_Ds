# Documentation for `src/core/evaluation.py`

## Overview
==========================================================
 Script: evaluation.py
 Author: M. El Aabaribaoune (@um6p)
 Description:
     Model evaluation and diagnostics.

     - Loads trained model
     - Runs inference on test data
     - Saves predictions to NetCDF
     - Generates diagnostic plots

 Notes:
     - GPU-safe (chunked inference)
     - Model-agnostic (ViT by default)
==========================================================

## Functions
### `def _build_model(...)`
No description.

### `def evaluate_and_save(...)`
No description.

## How to Modify
If you need to make changes to `evaluation.py`:
- **Logic changes**: Locate the corresponding function. Update its docstring if you change its signature or behavior.
- Ensure any related imports in other files are updated if you change function/class names.