#!/bin/bash
#SBATCH --job-name=insitu_diag
#SBATCH --output=/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/logs/insitu_diag_%j.out
#SBATCH --error=/srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/logs/insitu_diag_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
source ~/.bashrc
conda activate clean_env_Pytorch

cd /srv/data/mohammad.elaabaribao/work/papers/downscaling/postproc/src/insitu

# ======================================================================
# FLAGS POUR CHOISIR LES FIGURES À GÉNÉRER (1 = oui, 0 = non)
# ======================================================================
RUN_NEAREST_PIXEL=1
RUN_QQPLOT=1
RUN_ANNUAL_CYCLE=1
RUN_TAYLOR_DIAGRAM=1
RUN_HEATMAPS=1
# ======================================================================

echo "========================================================"
echo " DÉBUT DU JOB IN-SITU : $(date)"
echo "========================================================"

if [ "$RUN_NEAREST_PIXEL" -eq 1 ]; then
    echo ""
    echo ">>> Génération de : nearestPointStaion.png"
    python plot_station_pixel_distance.py
fi

if [ "$RUN_QQPLOT" -eq 1 ]; then
    echo ""
    echo ">>> Génération de : qqplot_insitu.png"
    python plot_qq_quantiles.py
fi

if [ "$RUN_ANNUAL_CYCLE" -eq 1 ]; then
    echo ""
    echo ">>> Génération de : annual_cycle_insitu.png"
    python plot_annual_cycle.py
fi

if [ "$RUN_TAYLOR_DIAGRAM" -eq 1 ]; then
    echo ""
    echo ">>> Génération de : taylor_diagram_insitu.png"
    python plot_taylor_diagram.py
fi

if [ "$RUN_HEATMAPS" -eq 1 ]; then
    echo ""
    echo ">>> Génération des Heatmaps (RMSE, Correlation, Bias, Combined)"
    python plot_metrics_heatmap.py
fi

echo ""
echo "========================================================"
echo " FIN DU JOB IN-SITU : $(date)"
echo "========================================================"
