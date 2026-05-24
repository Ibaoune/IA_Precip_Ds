<!-- Author: M. El Aabaribaoune (@um6p) -->
# Documentation for `src/models/glm.py`

## Overview
No module-level description provided.

## Classes
### `class PixelWiseGLM`
Optimized Pixel-Wise GLM for Climate Downscaling.

Features:
- Bernoulli-Gamma implementation for precipitation.
- Gaussian implementation for temperature.
- Matrix-vectorized prediction (orders of magnitude faster than pixel-wise loops).
- Lightweight storage (only saves model coefficients, not full stat-objects).

**Methods:**
- `__init__(...)`: No description.
- `predict(...)`: Matrix-vectorized prediction.

## Functions
### `def _train_pixel(...)`
Helper for parallel execution. Fits one pixel.

### `def train_glm(...)`
Parallelized GLM trainer.

## How to Modify
If you need to make changes to `glm.py`:
- **Architecture/Object changes**: Look into the classes defined above. Add or modify methods as needed.
- **Logic changes**: Locate the corresponding function. Update its docstring if you change its signature or behavior.
- Ensure any related imports in other files are updated if you change function/class names.