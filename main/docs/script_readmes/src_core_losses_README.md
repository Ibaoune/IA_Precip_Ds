# Documentation for `src/core/losses.py`

## Overview
No module-level description provided.

## Classes
### `class BernoulliGammaLoss`
No description.

**Methods:**
- `__init__(...)`: No description.
- `forward(...)`: pred: (B, 3, H, W) - [occurrence, shape, scale]

### `class GaussianLoss`
No description.

**Methods:**
- `__init__(...)`: No description.
- `forward(...)`: pred: (B, 2, H, W) - [mean, log_var]

## How to Modify
If you need to make changes to `losses.py`:
- **Architecture/Object changes**: Look into the classes defined above. Add or modify methods as needed.
- Ensure any related imports in other files are updated if you change function/class names.