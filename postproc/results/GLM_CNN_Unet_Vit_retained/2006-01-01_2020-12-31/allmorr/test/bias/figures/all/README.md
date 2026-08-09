<!-- Author: M. El Aabaribaoune (@um6p) -->
# Documentation des Figures : Biais & Moyenne

Ce répertoire contient les figures de haute qualité évaluant les modèles de descente d'échelle (downscaling) par rapport aux observations (MSWEP) pour la période annuelle (`Annual`).

## Processus et Origine des Données (Pipeline)

Les figures générées s'appuient sur les sorties brutes des quatre architectures de modèles de downscaling retenues (`GLM`, `CNN`, `U-Net`, `ViT`) qui descendent l'échelle des données ERA5 vers la résolution MSWEP.

**1. Données Brutes Originales (Downscaling) :**
Les prédictions journalières originales sont issues des inférences du modèle, stockées sous le format `{Modèle}_predictions_era5_to_mswep.nc` :
- `GLM` : `main/results/glm/retained/.../glm_predictions_era5_to_mswep.nc`
- `CNN` : `main/results/cnn/retained/.../cnn_predictions_era5_to_mswep.nc`
- `U-Net` : `main/results/unet/tests/.../unet_exp32_predictions_era5_to_mswep.nc`
- `ViT` : `main/results/vit/retained/.../vit_predictions_era5_to_mswep.nc`

**2. Calcul des Métriques (Post-Processing) :**
Ces données brutes ont ensuite été traitées par le framework d'évaluation (orchestrateur `postproc/eval.py` exécutant `postproc/src/mean/bias/Bias.py`, avec les paramètres définis dans `postproc/configs/config_final_retained_models.yaml`). Le traitement a appliqué les étapes suivantes :
- **Extraction temporelle** : Filtrage strict sur la période de test indépendante `[2006-01-01, 2020-12-31]`.
- **Masquage spatial** : Application d'un masque terre/mer et des frontières pour se restreindre uniquement au domaine terrestre du Maroc (`region: allmorr`, `mask_land: true`, `only_morocco: true`).
- **Agrégation** : Calcul des moyennes temporelles climatologiques et des erreurs (Biais) à l'échelle annuelle (`Annual`).
- **Sortie intermédiaire** : Les résultats sont stockés dans des fichiers NetCDF légers contenant directement les cartes de biais et de moyenne :
  `postproc/results/GLM_CNN_Unet_Vit_retained/2006-01-01_2020-12-31/allmorr/test/bias/results/{Modèle}_pr_allmorr_calcul_land_strategy_mean_first_corr_per_year_Annual.nc`

**3. Données de Référence (Observation) :**
- L'observation MSWEP (1979-2020) a été chargée pour comparaison depuis :
  `/srv/lustre01/project/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/obs/mswep/mswep_1979_2020.nc`

## Scripts de Génération Reproductible
Pour garantir un positionnement parfait (usage strict de `GridSpec`) et des colorbars normalisées (intervalle centré blanc pour le biais), des scripts dédiés ont été utilisés en contournant le pipeline générique :

- **Figure Combinée** (`mean_precip_and_spatial_bias_error_Annual.png`) : Générée par `postproc/plot_paper_mean_bias.py`
- **Figures Individuelles** (`mean_precipitation_Annual.png`, `spatial_bias_error_Annual.png`) : Regénérées avec les mêmes standards esthétiques (colorbars exactes et polices augmentées) via le script `postproc/plot_individual_figures.py`

## Figures Générées

### 1. `mean_precipitation_Annual.png`
- **Description** : Comparaison de la précipitation moyenne annuelle de chaque modèle avec l'observation de référence (MSWEP).
- **Formule** : Moyenne temporelle des précipitations journalières sur toute la période, soit $\frac{1}{N} \sum_{t=1}^{N} P(t)$.
- **Unité** : `mm/day`
- **Interprétation** : Permet d'observer visuellement si le modèle reproduit bien la distribution spatiale climatologique (par exemple, les fortes pluies sur le Rif et le Moyen Atlas).

### 2. `spatial_bias_error_Annual.png`
- **Description** : Carte de l'erreur systématique (biais) spatiale par rapport à MSWEP.
- **Formule** : Biais = (Précipitation Moyenne du Modèle) - (Précipitation Moyenne MSWEP).
- **Unité** : `mm/day`
- **Interprétation** :
  - **Couleurs Bleues** : Le modèle **surestime** la précipitation (biais positif).
  - **Couleurs Rouges** : Le modèle **sous-estime** la précipitation (biais négatif).
  - **Blanc pur (`[-0.1, 0.1]`)** : Le modèle reproduit presque parfaitement la réalité (biais négligeable).

### 3. `mean_precip_and_spatial_bias_error_Annual.png`
- Version composite finale (panels (a) et (b)) réunissant les deux figures précédentes sur une seule grille mathématiquement centrée.

## Colorbars & Esthétisme
- **Biais** : Échelle divergente `RdBu` discrétisée aux bornes strictes `[-2, -1, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1, 2]`. La zone centrale `[-0.1, 0.1]` est volontairement blanchie pour épurer visuellement la carte.
- **Moyenne** : Échelle `custom_precip` originelle aux bornes `[0, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4, 5, 6, 8, 10]`, avec le premier intervalle `[0, 0.1]` en blanc pour distinguer clairement les zones sèches des pluies faibles.
