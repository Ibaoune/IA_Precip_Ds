<!-- Author: M. El Aabaribaoune (@um6p) -->
# Pipeline d'Évaluation de la Descente d'Échelle (Downscaling) Climatologique & des Précipitations Extrêmes

Ce répertoire (`postproc/`) contient la logique centralisée du post-traitement, de l'évaluation et de la visualisation pour la descente d'échelle statistique (downscaling) des précipitations sur le Maroc. Il permet le calcul des métriques climatologiques standards ainsi que l'évaluation des indices de précipitations extrêmes, en comparant les prédictions mises à l'échelle (provenant de modèles comme GLM, CNN, U-Net, et ViT) aux observations (MSWEP) et aux modèles de circulation générale (GCM) bruts (LMDZ) servant de référence.

---

## Logique et Structure du Répertoire

La logique de ce répertoire repose sur la séparation des responsabilités : configuration, préparation des données, code source (calculs) et exécution. Voici le rôle détaillé de chaque composant :

* **`configs/`** : Fichiers de configuration (YAML) qui définissent les scénarios d'évaluation, les limites spatiales et temporelles, et les métriques à activer. Cela permet d'exécuter différentes évaluations sans avoir à modifier le code Python.
  * On y trouve divers scénarios : comparaisons à fine résolution (`config_infer_lmdz250_to_mswep.yaml`), comparaisons directes à la grille brute GCM (`config_infer_lmdz250_to_raw10km.yaml`), comparaisons croisées de scénarios (`config_scenarios_comparison.yaml`), ainsi qu'un sous-dossier `tests/` pour les configurations d'essai.
* **`datasets/`** : Cache local pour les bases de références GCM alignées, converties en termes d'unités (vers mm/jour) et remises à l'échelle (grossies ou affinées par interpolation bilinéaire).
* **`src/`** : Le cœur du code source Python gérant le pipeline.
  * `postproc.py` : Script maître (orchestrateur) qui lance le calcul de toutes les métriques pour les régions actives.
  * `utils.py` : Fonctions utilitaires partagées (masquage géographique, palettes de couleurs standardisées, renommage pour l'affichage).
  * **Scripts spécialisés** : Scripts ciblés pour générer les figures de l'article scientifique (`plot_lmdz_paper_figures.py`), les résumés saisonniers ou régionaux (`seasonal_summary.py`, `regional_summary.py`), ou évaluer l'apport de configurations précises (`plot_scenario2_added_value.py`).
  * **Sous-dossiers (`mean/`, `extreme/`, etc.)** : Contiennent la logique de calcul spécialisée pour chaque métrique.
* **`scripts/`** : Scripts bash d'aide à l'exécution et à l'exploration.
  * **`jobs/`** : Contient les templates et scripts de soumission Slurm (`.sh`) pour lancer les calculs sur cluster.
  * **`logs/`** : Stocke les logs d'exécution de ces jobs.
  * **Outils d'exploration** : Scripts locaux (`explore.sh`, `evaluate.sh`) et outils python annexes (`generate_region_pdfs.py`).
* **`shape_files/`** : Contient les masques géographiques (fichiers `.shp`) pour le Maroc et ses sous-régions (Nord, Sud, Est, Nord-Est), ce qui permet des évaluations spatialement ciblées.
* **`results/`** : Répertoire de sortie structuré hiérarchiquement pour organiser les résultats, les cartes générées, et les données NetCDF : `results/<experiment_name>/<dates>/<region>/`.
* **`logs/`** : Fichiers de journalisation (logs) générés par les exécutions pour suivre le bon fonctionnement des tâches ou déboguer.
* **`insitu/`** : Répertoire destiné à l'évaluation spécifique ou à la préparation des données provenant de stations météorologiques terrestres in situ (données d'observation locales).
* **`docs/`** : Documentation supplémentaire du projet.

### Fichiers Importants à la Racine du Répertoire
* **`README_METRICS.md`** : Documentation exhaustive expliquant *comment* et *pourquoi* chaque métrique (Biais, RMSE, CDD, R95, ROCSS, etc.) est calculée (logique mathématique et interprétation).
* **`prepare_comparison_datasets.py`** : Script de préparation automatisé. Il convertit les unités, normalise les repères temporels et gère l'interpolation spatiale des données brutes GCM (LMDZ) pour qu'elles soient prêtes à être comparées.
* **`generate_all_pdfs.py`** : Script utilitaire qui parcourt les dossiers de résultats pour rassembler et concaténer toutes les figures (images `.png`) générées en un seul rapport global au format PDF (par exemple `all_figures_summary.pdf`) par scénario.
* **`glue_scenarios.py`** : Script qui "colle" (fusionne) spatialement différentes prédictions NetCDF régionales en utilisant les masques géographiques (shapefiles) afin de reconstituer une carte globale composite pour l'ensemble du pays.
* **`submit_all_postproc.sh`** : Script maître Bash facilitant la soumission en bloc de multiples tâches de post-traitement sur le cluster Slurm, évitant de lancer chaque fichier `.sh` individuellement.

---

## Pipeline de Préparation et d'Alignement des Données

Les modèles de circulation générale (GCM) produisent souvent des données sur des grilles très grossières et avec des unités physiques hétérogènes. Le script [`prepare_comparison_datasets.py`](./prepare_comparison_datasets.py) uniformise tout cela automatiquement :

1. **Conversion des Unités Journalières** : Les sorties brutes GCM expriment souvent les précipitations en flux de densité (ex: `kg/(s*m2)`). Le script les convertit en épaisseur journalière (**`mm/jour`**) en multipliant par **`86400.0`**, ce qui correspond aux unités MSWEP.
2. **Normalisation Temporelle** : Les données journalières brutes LMDZ sont marquées à midi (`12:00:00`), tandis que les prédictions (downscaling) et MSWEP sont moyennées sur la journée avec un repère à minuit. Le script arrondit les repères temporels à minuit (grâce à `dt.floor('D')`).
3. **Mappage Spatial Bi-directionnel** :
   * **Interpolation Bilinéaire** : Re-échantillonne les grilles GCM grossières vers la résolution fine de 10km (pour se comparer aux prédictions et MSWEP).
   * **Grossissement (Coarsening)** : Dégrade la résolution fine des prédictions (downscaling) vers la grille native du modèle parent GCM pour permettre des évaluations directes à basse résolution ("rough" grid).

**Pour l'exécuter :**
```bash
python3 prepare_comparison_datasets.py
```

---

## Esthétique Climatologique Standardisée

Le fichier de base `src/utils.py` impose des règles strictes de formatage pour garantir une esthétique professionnelle, lisible et uniforme dans toutes les visualisations :

* **Nomenclature Standard (`get_display_name`)** : Traduit dynamiquement les noms internes des variables et des modèles pour les titres, axes et légendes (ex. affiche proprement "U-Net" ou "MSWEP").
* **Couleurs Persistantes (`GLOBAL_MODEL_COLORS`)** : Assure qu'un modèle donné possède **toujours la même couleur** sur tous les graphiques (cartes, boxplots, séries temporelles) :
  * **`MSWEP`** (Observation de référence) : `#000000` (Noir)
  * **`LMDZ 250`** : `#F4A261` (Orange chaud)
  * **`LMDZ 3.5`** : `#E76F51` (Rouge-Orange chaud)
  * **`GLM`** : `#2A9D8F` (Bleu sarcelle/Vert)
  * **`CNN`** : `#457B9D` (Bleu clair)
  * **`U-Net`** : `#1D3557` (Bleu foncé)
  * **`ViT`** : `#E63946` (Rouge vif)

---

## Guide d'Exécution

Assurez-vous que votre environnement Conda est activé avant de lancer les tâches :
```bash
conda activate clean_env_Pytorch
```

### Exécution Locale
Vous pouvez lancer manuellement le post-traitement pour une configuration spécifique directement en ligne de commande :
```bash
python3 -u src/postproc.py configs/config_infer_lmdz250_to_mswep.yaml
```

### Soumission sur Cluster SLURM
Le répertoire `scripts/` (et le fichier `submit_all_postproc.sh`) permettent de lancer les calculs de manière distribuée.

*Vous pouvez utiliser un template d'exécution avec une configuration ciblée :*
```bash
# Exemple : Comparer les modèles descendants LMDZ250 aux observations MSWEP
sbatch scripts/job_postproc_infer_template.sh configs/config_infer_lmdz250_to_mswep.yaml
```
*(Ou bien simplement exécuter `./submit_all_postproc.sh` pour soumettre les différentes évaluations d'un seul coup).*
