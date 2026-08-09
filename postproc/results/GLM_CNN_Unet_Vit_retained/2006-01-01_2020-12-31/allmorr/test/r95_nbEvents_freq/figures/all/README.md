<!-- Author: M. El Aabaribaoune (@um6p) -->
# Documentation des Figures : Événements Extrêmes (Fréquence R95)

Ce répertoire contient les figures de haute qualité évaluant la capacité des modèles de descente d'échelle (downscaling) à reproduire les extrêmes de précipitations (fréquence des événements > 95ème percentile) par rapport aux observations (MSWEP).

## Processus et Origine des Données (Pipeline)

Les figures générées s'appuient sur les sorties brutes des modèles (`GLM`, `CNN`, `U-Net`, `ViT`) qui descendent l'échelle des données ERA5 vers MSWEP.

**1. Données Brutes Originales (Downscaling) :**
Les prédictions journalières de précipitations sont stockées sous le format `{Modèle}_predictions_era5_to_mswep.nc`.

**2. Calcul des Métriques Extrêmes (Post-Processing) :**
Les données ont été traitées par le framework d'évaluation (orchestrateur `postproc/eval.py` exécutant `postproc/src/extreme/indices.py` ou module équivalent, configuré par `postproc/configs/config_final_retained_models.yaml`). Le traitement effectue :
- **Extraction temporelle** : Filtrage sur la période de test `[2006-01-01, 2020-12-31]`.
- **Masquage spatial** : Application du masque terrestre du Maroc (`region: allmorr`).
- **Calcul R95** : Détection des événements dépassant le 95ème percentile, calculé de façon dynamique ou via un seuil climatologique, puis agrégation du nombre de jours (fréquence) par année.
- **Sortie intermédiaire** : Les cartes annuelles d'extrêmes sont sauvegardées sous le format :
  `postproc/results/GLM_CNN_Unet_Vit_retained/2006-01-01_2020-12-31/allmorr/test/r95_nbEvents_freq/results/{Modèle}_pr_allmorr_calcul_land_strategy_daily_first_corr_per_year_Annual.nc`

**3. Données de Référence (Observation) :**
L'observation MSWEP a subi le même traitement d'extraction R95 et est disponible dans le fichier précalculé correspondant.

## Scripts de Génération Reproductible
Afin d'assurer un formatage strict (grille `GridSpec` centrée, colorbars standardisées selon la configuration originelle), les figures ont été générées avec des scripts dédiés :

- **Figure Combinée** (`r95_nbEvents_freq_spatial_map_and_error_Annual.png`) : Générée par `postproc/plot_paper_r95_freq.py`
- **Figures Individuelles** (`r95_nbEvents_freq_spatial_map_Annual.png`, `spatial_frequency_error_Annual.png`) : Générées via le script `postproc/plot_individual_r95_freq.py`

## Figures Générées

### 1. `r95_nbEvents_freq_spatial_map_Annual.png`
- **Description** : Comparaison de la fréquence spatiale des événements extrêmes R95 de chaque modèle avec MSWEP.
- **Unité** : `Days/Year` (Jours par an)
- **Palette** : `custom_freq` (allant du blanc pour 0 événement jusqu'au magenta pour les fréquences très élevées).
- **Interprétation** : Permet de voir où les événements extrêmes sont les plus fréquents (généralement dans les zones montagneuses du Rif).

### 2. `spatial_frequency_error_Annual.png`
- **Description** : Carte du biais (erreur systématique) de la fréquence des événements R95 par rapport à MSWEP.
- **Formule** : Erreur = (Fréquence du Modèle) - (Fréquence MSWEP).
- **Unité** : `Days/Year`
- **Palette** : Divergente classique (`RdBu`), identique à la configuration de base.
- **Interprétation** :
  - **Couleurs Bleues** : Le modèle génère trop d'événements extrêmes (surestimation).
  - **Couleurs Rouges** : Le modèle rate des événements extrêmes (sous-estimation).
  - **Couleurs Blanches** : Biais négligeable.

### 3. `r95_nbEvents_freq_spatial_map_and_error_Annual.png`
- Version composite réunissant les deux figures précédentes sur une grille propre, prête pour l'intégration au manuscrit de publication.
