<!-- Author: M. El Aabaribaoune (@um6p) -->
# Regional & Macro-Region Loss Masking Guide (Scenario 3)

This guide documents the **Regional Loss Masking** methodology implemented in the downscaling framework. This setup allows models to be trained over the **full domain** (preserving complete spatial boundary conditions) while calculating optimization loss and gradient updates exclusively on specific sub-regions.

---

## 💡 Methodology & Motivation

When deep learning models (such as CNNs and Vision Transformers) are trained on small, cropped regional grids:
1. **Edge Effects / Artifacts**: They suffer from boundary discontinuities because they lack spatial context immediately outside the crop border.
2. **Missing Large-scale Context**: Global features (like atmospheric moisture transport from the Atlantic or Saharan winds) cannot be fully captured because self-attention or convolutions are restricted to the crop window.

### The Loss Masking Solution
To resolve this, we train the models on the **entire Moroccan domain** but mask the loss calculations:
*   The model receives full-domain inputs and predicts full-domain outputs.
*   During backpropagation, the loss function (e.g., Bernoulli-Gamma Loss) is multiplied by a binary spatial mask:
    $$\text{Loss}_{\text{masked}} = \frac{\sum (\text{Loss}_{\text{element-wise}} \times \text{Mask})}{\text{Batch Size} \times \sum \text{Mask}}$$
*   Gradients are only computed and propagated back from pixels inside the active mask, focusing the model's parameter tuning exclusively on that region's dynamics.

---

## 🗺️ Supported Regions & Macro-Regions

The system reads shapefiles directly from the `postproc/shape_files/` directory.

### 1. Individual Sub-Regions
*   **`north`**: Covers northern mountains and Mediterranean-influenced areas.
*   **`north_east`**: Covers northeast regions.
*   **`east`**: Covers east semi-arid areas.
*   **`south`**: Covers dry Saharan provinces.

### 2. Scenario 3 Macro-Regions
For Scenario 3, shapefiles are merged together to represent larger climatological regimes:
*   **`north_northeast`** (Northern / semi-humid regime): Merges `north.shp` and `north_east.shp`.
*   **`east_south`** (Southern / dry regime): Merges `east.shp` and `south.shp`.

---

## ⚙️ Configuration (YAML Syntax)

To enable loss masking, configure the `loss_mask` block under the `training` section of your YAML file.

### Single Sub-Region Example
```yaml
training:
  loss_mask:
    enable: true
    region: north
    shapefile: postproc/shape_files/north.shp
```

### Merged Macro-Region Example (Scenario 3)
```yaml
training:
  loss_mask:
    enable: true
    region: north_northeast
    shapefile:
      - postproc/shape_files/north.shp
      - postproc/shape_files/north_east.shp
```

---

## 🛠️ Internal Processing Pipeline

When `loss_mask.enable: true` is configured:
1.  **Config Parser (`config.py`)**:
    *   Reads `loss_mask.shapefile` (supports string path or lists).
    *   Resolves relative paths to absolute paths using `root_dir`.
    *   Appends `_lossmask_<region_name>` to the experiment name to separate model checkpoints and logs automatically (unless the experiment name already ends with the region name).
2.  **Mask Generator (`train.py`)**:
    *   Loads shapefile(s) using `geopandas`.
    *   If a list of shapefiles is provided, merges them using `pandas.concat` and dissolves them into a single geometry.
    *   Rasterizes the geometry onto the output grid coordinates (`lat_out`, `lon_out`).
    *   Multiplies it by the Natural Earth land-sea mask to ensure no loss is computed over the ocean.
    *   Converts the combined mask to a PyTorch tensor and passes it to `train_model()`.

---

## 🚀 Job Generation & Execution

Generators are provided to automatically build configs and SLURM jobs:

### 1. Sub-Region Loss Masking
Generate 12 configurations (3 models $\times$ 4 sub-regions) by running:
```bash
python generate_lossmask_setups.py
```
Submit jobs:
```bash
sbatch scripts/loss_mask/job_gpu_<model_name>_lossmask_<region>.sh
```

### 2. Macro-Region Loss Masking (Scenario 3)
Generate 6 configurations (3 models $\times$ 2 macro-regions) by running:
```bash
python generate_scenario3_setups.py
```
Submit jobs:
```bash
sbatch scripts/scenario3/job_gpu_<model_name>_scenario3_<region>.sh
```
