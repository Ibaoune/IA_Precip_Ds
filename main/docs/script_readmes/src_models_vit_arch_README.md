<!-- Author: M. El Aabaribaoune (@um6p) -->
# Documentation for `src/models/vit_arch.py`

## Overview
No module-level description provided.

## Classes
### `class BernoulliGammaLoss`
Custom loss function tailored for precipitation downscaling.
Models rainfall as a mixed discrete-continuous process:
- Bernoulli distribution for rain occurrence.
- Gamma distribution for rainfall intensity.

**Methods:**
- `__init__(...)`: No description.
- `forward(...)`: No description.

### `class PatchEmbedding`
Splits the 2D input grid into non-overlapping patches and projects 
them into a linear embedding space.

**Methods:**
- `__init__(...)`: No description.
- `forward(...)`: No description.

### `class TransformerBlock`
A single Vision Transformer block featuring Multi-Head Self Attention 
and a Feed-Forward Network with GELU activation.

**Methods:**
- `__init__(...)`: No description.
- `forward(...)`: No description.

### `class TransformerEncoder`
Stacks multiple Transformer blocks to form the encoder.

**Methods:**
- `__init__(...)`: No description.
- `forward(...)`: No description.

### `class UpsamplingDecoder`
Reconstructs the spatial grid from the encoded sequence of patches 
using transposed convolution, and pads to the exact target dimensions if necessary.

**Methods:**
- `__init__(...)`: No description.
- `forward(...)`: No description.

### `class DownscalingViT`
Complete Vision Transformer for 2D data downscaling.
Transforms low-resolution inputs into high-resolution targets.

**Methods:**
- `__init__(...)`: No description.
- `forward(...)`: x shape expected: (batch_size, in_channels, height, width)

## How to Modify
If you need to make changes to `vit_arch.py`:
- **Architecture/Object changes**: Look into the classes defined above. Add or modify methods as needed.
- Ensure any related imports in other files are updated if you change function/class names.