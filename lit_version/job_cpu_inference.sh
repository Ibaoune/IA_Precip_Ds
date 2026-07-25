#!/bin/bash

#SBATCH --job-name=inferences_vit        # Job Name
#SBATCH --output=out_%j.log        # Everything (stdout & stderr) goes here
#SBATCH --error=out_%j.log         # Can also merge stderr with stdout
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU


# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Record start time
start_time=$(date +%s)

# Run main script
# stdout and stderr are automatically redirected to the SLURM output file

ppython train.py --config config_mse_global_x.yaml --model vit --inference
python train.py --config config_mse_global_y.yaml --model vit --inference
python train.py --config config_mse_log.yaml --model vit --inference
python train.py --config config_mse_no_norm.yaml --model vit --inference
python train.py --config config_mse_per_channel.yaml --model vit --inference
python train.py --config config_mse_per_day.yaml --model vit --inference
python train.py --config config_mse_log_global_y.yaml --model vit --inference
python train.py --config config_nll_global_x.yaml --model vit --inference
python train.py --config config_nll_log.yaml --model vit --inference
python train.py --config config_nll_no_norm.yaml --model vit --inference
python train.py --config config_nll_per_channel.yaml --model vit --inference
python train.py --config config_nll_per_day.yaml --model vit --inference

# Record end time and log total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."

