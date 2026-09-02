# In-Situ Validation and Post-Processing

**Author:** M. El Aabaribaoune (@um6p)

This directory contains the Python scripts and configurations for the validation of regional climate downscaling models against in-situ station observations. 

## Design Philosophy

The pipeline is designed to be highly modular and configuration-driven. The configurations dictate the input paths, station definitions, analysis periods, and output locations. The Python scripts focus purely on data processing, metric computation, and visualization.

## Configuration

The processing is driven by a single YAML configuration file.

- **config_evaluation.yaml**: Main configuration file containing the reference to the observational data, gridded climate model outputs (GLM, CNN, U-Net, ViT), the retained meteorological stations, the analysis period, and the output directory.

All python scripts automatically load parameters from this file, eliminating hardcoded paths or station lists. 

## Scripts and Diagnostics

The diagnostic scripts in this directory each serve a specific evaluation purpose:

### Utility Modules
- **utils_insitu.py**: Provides common helper functions such as data loading, coordinate matching (finding nearest valid grid points), time series alignment, and metric computation.

### Diagnostic Visualizations
All diagnostic scripts are configured to only output publication-ready, high-resolution `.png` figures. Temporary tabular data and `.pdf` files are intentionally excluded from generation to maintain a clean workspace.

- **plot_station_pixel_distance.py**: Visualizes the spatial relationship between the exact station coordinates and the nearest valid gridded data pixel selected for analysis. Generates a map.
- **plot_multimodel_gridpoints.py**: Generates a unified spatial diagnostic map showing the grid points selected by all models for each station, handling potential fallbacks due to missing data (NaNs).
- **plot_annual_cycle.py**: Computes and plots the monthly climatological annual cycle for both observations and simulated model estimates.
- **plot_climatology.py**: A robust climatological processor focusing on extracting long-term station averages and visualizing them.
- **plot_qq_quantiles.py**: Generates Quantile-Quantile (Q-Q) plots on a logarithmic scale to evaluate the models' ability to capture the distribution of daily precipitation, particularly extreme events.
- **plot_taylor_diagram.py**: Renders a Taylor diagram to visually synthesize the correlation, root-mean-square error (RMSE), and standard deviation between the models and the observations.
- **plot_metrics_heatmap.py**: Generates station-wise performance heatmaps covering deterministic metrics like RMSE, Temporal Correlation, and Wet-day Frequency Bias.
- **plot_overall_validation.py**: A comprehensive aggregation script that creates a multi-panel summary figure combining spatial station locations and overall metric heatmaps.

## Execution

The execution of these scripts is automated through the SLURM workload manager. 

Do not execute these scripts manually unless debugging. For production processing, use the dedicated SLURM job file located at:
`../../slurm/insitu/job_insitu.sh`

The `job_insitu.sh` script provides toggle flags to selectively run specific diagnostics without rebuilding the entire pipeline.
