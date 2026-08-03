# Post-Processing Scripts Repository

**Author:** M. El Aabaribaoune (@um6p)

This directory contains the core execution scripts, orchestration logic, and utility functions required to run the post-processing and evaluation pipeline for the downscaling models. The pipeline is responsible for computing climatological metrics (both mean state and extremes) and generating publication-quality figures based on the model predictions.

## Directory Structure and Contents

### Directories
- **`jobs/`**: Contains the SLURM batch submission scripts used to execute the post-processing tasks on the High-Performance Computing (HPC) cluster. The final, active job script (`job_final_retained_models.sh`) is located here, while obsolete or experimental scripts have been cleanly archived into `jobs/others/`.
- **`utils/`**: Contains shared utility modules and helper functions imported by the main Python scripts (e.g., custom colorbars, spatial mapping tools, statistical metric calculators).
- **`logs/`**: The designated output directory for SLURM `.out` and `.err` log files generated during the execution of post-processing jobs.

### Core Scripts
- **`setup_and_run_experiments_postproc.py`**: A comprehensive Python orchestrator script designed to parse experiment configurations, prepare the directory structures, and automate the execution of multiple post-processing tasks sequentially or in parallel.
- **`generate_region_pdfs.py`**: A Python script responsible for aggregating the computed spatial and temporal metrics and compiling them into clean, standardized PDF reports tailored for specific regions.
- **`submit_all_postproc.sh`**: A shell script utility to rapidly submit a batch of post-processing SLURM jobs to the cluster queue at once.
- **`evaluate.sh` / `explore.sh`**: Auxiliary bash scripts used for quick preliminary evaluation and data exploration of the generated `.nc` prediction files before launching the full heavy pipeline.

## Usage
To execute the definitive post-processing pipeline for the final retained architectures (GLM, CNN, U-Net, ViT), navigate to the root directory of the repository and submit the corresponding SLURM job:
```bash
sbatch postproc/scripts/jobs/job_final_retained_models.sh
```
