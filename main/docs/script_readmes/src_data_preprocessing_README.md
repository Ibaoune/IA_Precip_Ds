# Documentation for `src/data/preprocessing.py`

## Overview
==========================================================
 Script: preprocessing.py
 Author: M. El Aabaribaoune (@um6p)
 Description:
     Data preprocessing utilities:
     - Normalization of predictors
     - Precipitation unit conversion
     - Conversion to PyTorch tensors

 Notes:
     - CPU-only preprocessing (GPU handled in training loop)
     - Model-agnostic
==========================================================

## Functions
### `def _print_stats(...)`
No description.

### `def _detect_units_from_source(...)`
Detect precipitation units based on dataset SOURCE.

### `def _convert_units(...)`
No description.

### `def preprocess_data(...)`
No description.

## How to Modify
If you need to make changes to `preprocessing.py`:
- **Logic changes**: Locate the corresponding function. Update its docstring if you change its signature or behavior.
- Ensure any related imports in other files are updated if you change function/class names.