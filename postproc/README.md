# Post-Processing Pipeline

This directory contains the post-processing and evaluation pipeline for the precipitation downscaling project. It provides scripts for calculating various climatological and extreme precipitation metrics, generating plots, and initially exploring model outputs.

## Directory Structure

- `exploration.py` / `explore.sh` / `config_explore.yaml`: Scripts for initial dataset exploration. Generates preliminary climatology maps, monthly annual cycles, and intensity distributions.
- `postproc.py` / `evaluate.sh` / `job_postproc.sh` / `config.yaml`: The master runner and its configurations for evaluating specific precipitation metrics (mean bias, RMSE, correlation, and extreme indices like CDD, r01, r95, r99, etc.).
- `utils.py`: Contains common utility functions for processing datasets (masking, calculating temporal evolutions, spatial mapping, etc.).
- `precip/`: Contains the specific Python scripts for individual metric calculations and their respective plotting scripts.
- `results/`: Directory where all generated plots and calculated metric data files are saved. Outputs are highly organized using the pattern: `results/<experiment_name>/<start_date>_<end_date>/<region_name>/`.
- `shape file/`: Shapefiles used for geographical masking (e.g., restricting calculations to land masses or specific regions like Morocco).

## Workflows

### 1. Dataset Exploration

Use the exploration pipeline to get a baseline understanding of model outputs compared to reference observation data.

1. **Configure**: Edit `config_explore.yaml` to set your experiment name, target region, the reference dataset, and the model datasets you wish to compare.
2. **Run Locally**:
   ```bash
   ./explore.sh
   # or explicitly:
   python3 exploration.py --config config_explore.yaml
   ```
3. **Outputs**: Generates plots for the mean annual cycle, intensity probability density functions (both log and linear KDE), spatial climatologies, and daily domain mean evolution inside `results/<experiment>/explore/`.

### 2. Metric Evaluation

The main post-processing runner evaluates specific climate and extreme precipitation metrics based on configuration.

1. **Configure**: Edit `config.yaml` to specify:
   - Which metric categories (e.g., `mean`, `extreme`) and specific metrics (e.g., `bias`, `cdd`, `r95`) are enabled.
   - The time range (`start_date` and `end_date`).
   - The specific regions you want to process over in the `regions` list (e.g., `['allmorr', 'north', 'south', 'east', 'north_east']`). The runner will iterate over each automatically.
2. **Run Locally**:
   ```bash
   ./evaluate.sh
   # or explicitly:
   python3 postproc.py
   ```
3. **Run on SLURM Cluster**:
   ```bash
   sbatch job_postproc.sh
   ```
   *Note: Ensure the correct Conda environment is specified inside the job script before submission.*
4. **Outputs**: Runs the individual metric calculation and plotting scripts located in `precip/`. Results are systematically grouped into `results/<experiment_name>/<start_date>_<end_date>/<region_name>/` making it extremely easy to compare models across different time periods and localized subregions.
