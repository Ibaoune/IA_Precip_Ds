<!-- Author: M. El Aabaribaoune (@um6p) -->
# Documentation des Figures : Indice d'Extrême de Sécheresse (CDD - Consecutive Dry Days)

Ce répertoire contient les figures prêtes pour publication évaluant la capacité des modèles de descente d'échelle (downscaling) à reproduire la distribution spatiale des périodes sèches prolongées, quantifiées par l'indice **CDD (Consecutive Dry Days)**, en comparaison avec la référence observationnelle (MSWEP).

## Qu'est-ce qui est calculé ? (Métrique)

L'indice **CDD (Consecutive Dry Days)**, avec un seuil de `0.5 mm` (`cdd_0.5`), mesure le nombre maximum de jours consécutifs dans une année où les précipitations journalières sont strictement inférieures à 0.5 mm. Il s'agit d'un indicateur fondamental pour évaluer l'aridité, la durée des sécheresses et la variabilité interannuelle du climat au Maroc.

## Processus et Origine des Données (Pipeline)

Les figures sont générées à partir des sorties des modèles d'intelligence artificielle (`GLM`, `CNN`, `U-Net`, `ViT`) qui descendent l'échelle des réanalyses globales ERA5 pour matcher la résolution de MSWEP.

**1. Calcul de l'Indice CDD (Post-Processing) :**
Les séries temporelles de précipitations journalières issues des prédictions (modèles) et des observations (MSWEP) sont traitées par le framework d'évaluation (`postproc/eval.py`). Le pipeline exécute les opérations suivantes :
- **Extraction temporelle** : Filtre strict sur la période de test indépendante `[2006-01-01, 2020-12-31]`.
- **Masquage spatial** : Extraction restreinte aux mailles terrestres du domaine marocain (`region: allmorr`).
- **Calcul de l'Indice** : Détection des séquences continues de jours secs (Pr < 0.5 mm) et extraction de la plus longue séquence pour chaque année.
- **Stockage** : Les cartes annuelles de CDD sont sauvegardées sous le format NetCDF :
  `postproc/results/GLM_CNN_Unet_Vit_retained/2006-01-01_2020-12-31/allmorr/test/cdd/cdd_0.5/results/{Modèle}_pr_allmorr_calcul_land_strategy_per_year_corr_per_year_Annual.nc`

**2. Moyenne Climatologique :**
Pour les figures, une moyenne interannuelle de l'indice CDD est calculée sur les 15 ans de la période de test afin de dresser la climatologie de la sécheresse.

## Scripts de Génération Reproductible

Pour garantir une qualité visuelle irréprochable (contrôle strict des espaces via `GridSpec`, harmonisation des polices `fontsize=18/14` et palettes personnalisées), des scripts dédiés ont été créés pour remplacer la fonction de tracé générique :

- **Figure Combinée** (`spatial_cdd_comparison_and_error_Annual.png`) : Générée par `postproc/plot_paper_cdd.py`
- **Figures Individuelles** (`spatial_cdd_comparison_Annual.png`, `spatial_cdd_error_Annual.png`) : Générées par `postproc/plot_individual_cdd.py`

## Interprétation Visuelle

### 1. `spatial_cdd_comparison_Annual.png` (Ligne a)
- **Description** : Cartographie de la moyenne climatologique de la longueur maximale des périodes sèches (CDD).
- **Unité** : `Days` (Jours).
- **Palette** : Séquentielle `YlOrBr` (Jaune -> Orange -> Marron).
- **Interprétation** : Les zones marron foncé (e.g., régions sahariennes et sud) indiquent une très longue période sèche continue (aridité forte, > 300 jours). Les zones jaunes (e.g., le Rif, Moyen Atlas) indiquent des périodes sèches plus courtes.

### 2. `spatial_cdd_error_Annual.png` (Ligne b)
- **Description** : Biais spatial de l'indice CDD simulé par les modèles par rapport à MSWEP.
- **Formule** : Biais = (CDD Modèle) - (CDD MSWEP).
- **Unité** : `Days` (Jours).
- **Palette** : Divergente `RdBu` centrée sur zéro.
- **Adaptation Visuelle "Q1"** : Pour épurer la carte du bruit statistique, les biais considérés comme négligeables (intervalle strict `[-10, 10]` jours) sont forcés en **blanc pur**.
- **Interprétation** :
  - **Couleurs Rouges (Négatif)** : Le modèle sous-estime la durée des périodes sèches (trop humide).
  - **Couleurs Bleues (Positif)** : Le modèle surestime la durée des périodes sèches (trop sec).
  - **Couleurs Blanches** : Le modèle reproduit parfaitement (à ±10 jours près) la durée de la sécheresse observée par MSWEP.

### 3. `spatial_cdd_comparison_and_error_Annual.png`
- La version finale combinant de manière compacte l'état moyen et le biais pour les 4 modèles dans un seul panneau structuré (a et b), prêt à être inclus directement dans l'article de recherche.
