# Documentation for `src/core/config.py`

## Overview
==========================================================
 Script: config.py
 Author: M. El Aabaribaoune (@um6p)
 Description:
     Loads configuration from YAML and exposes attributes
     as a simple namespace with robust type casting
     (int / float / bool / string safe).
==========================================================

## Classes
### `class Config`
No description.

**Methods:**
- `__init__(...)`: No description.

## Functions
### `def _to_int(...)`
No description.

### `def _to_float(...)`
No description.

### `def _to_bool(...)`
No description.

### `def load_config(...)`
No description.

## How to Modify
If you need to make changes to `config.py`:
- **Architecture/Object changes**: Look into the classes defined above. Add or modify methods as needed.
- **Logic changes**: Locate the corresponding function. Update its docstring if you change its signature or behavior.
- Ensure any related imports in other files are updated if you change function/class names.