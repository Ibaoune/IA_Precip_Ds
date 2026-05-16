# PyTorch Lightning Downscaling Framework (`lit_version`)

This directory contains a legacy or alternative version of the precipitation downscaling models implemented using the **PyTorch Lightning** framework. It was designed to reduce boilerplate training code and easily scale across multiple GPUs.

## 📁 Directory Structure

- `train.py`: The main training script. It utilizes the `lit_module`, model architectures, and custom loss functions.
- `lit_module.py`: Defines the PyTorch Lightning classes:
  - `LitDataModule`: Handles data loading and connects to `dataset.DownscalingDataset`.
  - `LitModule`: Encapsulates the core training, validation logic, and optimizer stepping.
- `models/`: Contains the neural network architectures (U-Net, ViT) and loss functions (`losses.py`).
- `dataset.py` / `dataset1.py`: Custom PyTorch `Dataset` classes for loading NetCDF climate data.
- `plot_truthVSpred.py`: Utility script to visualize the truth versus predictions after inference.

## 🚀 Usage

### 1. Training a Model
To train a model, run the `train.py` script specifying the configuration and model type:
```bash
python train.py --config configs/config_mse.yaml --model unet
``` 
Weights and TensorBoard logs will be automatically saved into the `weights/` and `logs/` directories, respectively.

### 2. Monitoring Progress
This framework heavily relies on TensorBoard. Launch it to monitor validation loss and metrics:
```bash
tensorboard --logdir=logs
```

### 3. Inference
After a model is trained, you can run inference to generate `.nc` prediction files:
```bash
python train.py --config configs/config_mse.yaml --model unet --inference
```

### 4. Running on SLURM
Pre-configured scripts are available to launch jobs on a computing cluster:
- CPU Job: `sbatch job_cpu.sh`
- GPU Job: `sbatch job_gpu.sh`
- Inference Job: `sbatch job_cpu_inference.sh`
