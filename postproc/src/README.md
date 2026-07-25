<!-- Author: M. El Aabaribaoune (@um6p) -->
# Post-Processing Source Code (`postproc/src/`)

This directory contains the core evaluation and visualization engine de vos modèles de downscaling (U-Net, ViT, CNN, GLM) et de vos modèles climatiques globaux (LMDZ). 

All scripts are modularly organized to separate orchestration logic, metric calculations, and the generation of publication-ready scientific figures. pour séparer la logique d'orchestration, le calcul des métriques et la génération des figures destinées aux publications scientifiques.

## Directory Architecture

### Main Files
* **`postproc.py`** : The main entry point of the pipeline. It reads the YAML configuration file, loads the datasets, and orchestrates the execution of evaluation modules. Ce script lit votre fichier de configuration YAML, charge les jeux de données (prédictions `.nc` et observations MSWEP), et orchestre l'exécution des modules d'évaluation (`mean` et `extreme`).
* **`utils.py`** : The universal toolkit containing shared functions for data processing, statistical calculations, and plotting utilities. Ce fichier regroupe toutes les fonctions partagées pour le traitement des données (interpolation temporelle/spatiale, masquage), les calculs statistiques et les utilitaires de tracé (notamment la fonction de pagination affichant 6 cartes par figure).

### Metrics Calculation Modules
* **`mean/`** : Mean state evaluation.
  * *`bias/`* : Calculates annual and seasonal spatial error (Spatial Bias).
  * *`rmse/`* : Calculates Root Mean Square Error.
  * *`correlation/`* : Evaluates temporal and spatial correlation with observations.
* **`extreme/`** : Evaluation of extreme precipitation climate indices.
  * *`cdd`* (Consecutive Dry Days), *`r01`* (Wet days), *`r95`* / *`r99`* (Very/Extremely wet days), *`rocss`*, *`qqplot`*.

### Synthesis and Publication Tools
* **`summaries/`** : 
  * `regional_summary.py` et `seasonal_summary.py` : Scripts used to aggregate model performance by geographical sub-regions or by seasons to offer high-level comparisons. ou par saisons pour offrir une comparaison de haut niveau entre différentes architectures.
* **`paper_figures/`** : 
  * Contains custom plotting scripts used to generate final, polished figures ready for research articles. (ex: `plot_lmdz_paper_figures.py`, `plot_scenario2_added_value.py`) destinés à générer les figures finales, soignées et prêtes à être intégrées dans vos articles de recherche.

### Development and Tests
* **`exploration/`** : Workspace for exploratory analysis scripts and ad-hoc data queries. (`exploration.py`) et les requêtes ponctuelles sur les données.
* **`tests/`** : Validation scripts to ensure the numerical robustness of the post-processing functions. (ex: `test_limits.py`) pour garantir la robustesse numérique de vos fonctions de post-traitement.
* **`scratch/`** : Scratch directory for temporary tests or ephemeral working files. ou fichiers de travail éphémères.

---
**Execution Note :** All these scripts are typically called via the bash scripts located in `postproc/scripts/`, which handle SLURM and Python environment configuration before launching the main orchestrator. situés dans `postproc/scripts/` (ex: `job_postproc_retained.sh`), qui se chargent de configurer l'environnement SLURM et Python (`clean_env_Pytorch`) avant de lancer `postproc.py` ou les scripts de `summaries/`.
