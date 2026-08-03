# Post-Processing Scripts Repository

**Author:** M. El Aabaribaoune (@um6p)

This directory acts as the structural hub for the post-processing and evaluation execution pipeline. Following a massive repository cleanup to prepare for publication, all experimental and standalone orchestration scripts have been archived, leaving only the essential components required to reproduce the final climatological metrics.

## Directory Structure and Contents

### Tracked Directories
- **`jobs/`**: The core directory containing the SLURM batch submission scripts used to execute post-processing tasks on the High-Performance Computing (HPC) cluster. The definitive, active job script for the final retained architectures (`job_final_retained_models.sh`) is located here.
- **`utils/`**: Contains shared utility modules and helper functions imported by the main Python post-processing engine located in `src/` (e.g., custom colorbars, spatial mapping tools, specific metric calculators).
- **`logs/`**: The designated output directory for SLURM `.out` and `.err` log files generated during the execution of the jobs.

### Ignored Directories (Archived)
- **`others/`**: Contains all obsolete, experimental, and preliminary scripts that were historically located at the root of `scripts/` (such as `evaluate.sh`, `explore.sh`, `generate_region_pdfs.py`, and `setup_and_run_experiments_postproc.py`). These files are intentionally untracked by Git to maintain a pristine, publication-ready repository, but physically remain on the disk for local reference.
- **`jobs/others/`**: Similar to the above, this holds all past SLURM submission scripts used during the testing phase.

## Usage
To execute the definitive post-processing pipeline for the final retained architectures (GLM, CNN, U-Net, ViT), navigate to the root directory of the repository and submit the corresponding SLURM job:
```bash
sbatch postproc/scripts/jobs/job_final_retained_models.sh
```
