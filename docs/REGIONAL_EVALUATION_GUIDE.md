<!-- Author: M. El Aabaribaoune (@um6p) -->
# Regional Splits & Splicing Evaluation Guide

This guide documents the framework's capability to train and evaluate models across distinct regional boundaries, and how separate regional prediction outputs are assembled back into a single unified Morocco grid for post-processing comparison.

---

## Bounding Box Regional Split (North vs. South)

In addition to full-domain training, you can train models on specialized regional crops of the dataset:
* **Split Coordinate**: **`28.0°N`** is the boundary dividing the North and South regions of Morocco.
* **North Domain**: $28.0^\circ\text{N}$ to $37.0^\circ\text{N}$ (focuses on Mediterranean storms, Atlantic fronts, and Atlas mountain precipitation).
* **South Domain**: $21.0^\circ\text{N}$ to $28.0^\circ\text{N}$ (focuses on hyper-arid Saharan provinces).

To generate the configuration files and SLURM scripts for regional crops, execute:
```bash
python generate_regional_setups.py
```
This builds configs under `configs/<model>/regional/` and job scripts under `scripts/regional/`.

---

## Splicing Predictions (`glue_regional_predictions.py`)

To evaluate regional models using the standard post-processing metrics and plotting scripts, their separate regional outputs (North and South NetCDFs) must be spliced back together.

The tool **[`main/glue_regional_predictions.py`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/main/glue_regional_predictions.py)** handles this automatically:
1. Loads the separate NetCDF prediction datasets for the North and South runs of a model.
2. Splices them together along the latitude dimension at the $28.0^\circ\text{N}$ boundary.
3. Combines metadata and saves a single, unified Moroccan-domain NetCDF file under the `retained/` directory.

### Running the Splicing Tool
```bash
python glue_regional_predictions.py
```
*Note: Splicing is pre-configured for `cnn_exp3`, `cnn_exp5`, `vit_precip_exp21_best_hybrid`, and `glm_precip_l2`.*

---

## GLM Split Redundancy Property

During the implementation of regional splits, it was verified that spatial regional splitting is **completely redundant** for Generalized Linear Models (`PixelWiseGLM`). 

### Mathematical Proof
The GLM baseline model fits separate linear coefficients for each pixel $(i,j)$ completely independently of neighboring pixels:
$$\text{Precipitation}_{(i, j)} = \beta_0^{(i,j)} + \sum_{k} \beta_k^{(i,j)} \times \text{Predictor}_{k, (i, j)}$$

Because there is no spatial weight sharing or convolution operation, training a GLM on a cropped regional bounding box yields **mathematically identical coefficients** to training it on the full domain for all matching pixels.

As a result:
* Regional GLM training runs are unnecessary and excluded from evaluations.
* We simply crop the unified Morocco GLM predictions to evaluate its regional performance.

---

## Regional vs. Unified Comparison

Once regional predictions are glued back together, they can be directly compared against the unified models (which were trained on the entire domain at once).

A comparison setup is available under:
**[`postproc/configs/config_regional_vs_unified.yaml`](file:///srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/configs/config_regional_vs_unified.yaml)**

Submit the comparison post-processing job:
```bash
sbatch job_postproc_regional_vs_unified.sh
```
This generates spatial bias maps, boxplots, and seasonal extremes comparisons comparing:
1. **Unified Models**: Models trained globally.
2. **Glued Regional Models**: Models trained separately on North/South regions and spliced.
3. **Loss-Masked Models**: Models trained globally but optimized on specific regional losses.
