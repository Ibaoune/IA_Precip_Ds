#!/bin/bash
# Author: M. El Aabaribaoune (@um6p)

for i in {2..12}; do
 cat << INNER_EOF > eval_test_unet_exp${i}.sh
#!/bin/bash
#SBATCH --job-name=eval_unet_${i}
#SBATCH --output=logs/eval_unet_${i}_%j.log
#SBATCH --error=logs/eval_unet_${i}_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

source ~/.bashrc
conda activate clean_env_Pytorch

export PYTHONUNBUFFERED=1

echo "======================================"
echo "Evaluating UNet Experiment: ${i}"
echo "======================================"

cd ../../..
python3 -u eval.py configs/unet/tests/test_exp${i}.yaml

echo "======================================"
INNER_EOF

 if [ "$i" -eq 10 ]; then
 sbatch --dependency=afterok:7039880 eval_test_unet_exp${i}.sh
 else
 sbatch eval_test_unet_exp${i}.sh
 fi
done
