# Documentation for `src/models/unet_arch.py`

## Overview
No module-level description provided.

## Classes
### `class UNet`
No description.

**Methods:**
- `__init__(...)`: No description.
- `conv_block(...)`: Creates a block of TWO consecutive Convolution -> Norm -> ReLU sequences.
- `_match_size(...)`: Pad or crop upsampled tensor to match skip-connection spatial dims.
- `forward(...)`: No description.

## How to Modify
If you need to make changes to `unet_arch.py`:
- **Architecture/Object changes**: Look into the classes defined above. Add or modify methods as needed.
- Ensure any related imports in other files are updated if you change function/class names.