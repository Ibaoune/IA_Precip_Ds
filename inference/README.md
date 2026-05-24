# Moteur d'Inférence Unifié (Downscaling)

Ce dossier contient le moteur d'inférence universel et robuste pour déployer l'ensemble de vos modèles de descente d'échelle (ViT, CNN, UNet, GLM).

Le pipeline prend les poids entraînés sur la tâche historique (ERA5 vers MSWEP) et les applique sur de nouvelles périodes ou sur des projections climatiques de Modèles de Circulation Générale (GCMs, ex. LMDZ à différentes résolutions).

---

## 🌟 Fonctionnalités Clés et Robustesse

1. **Chargement Automatique de l'Architecture (`train_config_path`) :**
   Il vous suffit de renseigner le chemin vers le fichier `config.txt` issu de l'entraînement de votre modèle. Le script `predict.py` importe automatiquement les bons hyperparamètres (type de modèle, loss, dimensions de la grille, type d'interpolation).

2. **Alignement Spatial Automatique :**
   Le moteur lit les bornes spatiales (`lon_min/max`, `lat_min/max`) directement depuis la configuration d'entraînement du modèle. Cela garantit une correspondance parfaite des dimensions de la grille et évite tout conflit NetCDF.

3. **Correction de Biais Avancée (SDM) :**
   Lors du passage aux données GCM (LMDZ r35 ou r250), le module applique la méthode *Scaling Delta Mapping* pour corriger les biais systématiques des prédicteurs GCM par rapport à la climatologie historique d'ERA5 avant l'inférence.

4. **Contrôle Granulaire par Scénario (`enable: true/false`) :**
   Vous pouvez activer ou désactiver individuellement chaque jeu de données (ERA5, LMDZ r35, LMDZ r250) d'un simple flag dans la configuration.

---

## 🎛️ Structure du Fichier de Configuration de Référence (`config.yaml`)

Le fichier `config.yaml` sert de modèle maître pour tous vos tests. La section la plus importante est `prediction` :

```yaml
prediction:
  # 1. Pointez vers la configuration d'entraînement du modèle cible
  train_config_path: ../main/results/tests/glm/glm_precip_l2/.../config.txt
  
  # 2. Dossier de sortie pour ce modèle
  output_dir: results/output/glm_precip_l2

  # 3. Liste des scénarios à prédire
  scenarios:
  - name: era5_present
    enable: true         # Mettez 'false' pour ignorer ce jeu de données
    src: era5
    start: '2006-01-01'
    end: '2014-12-31'
    bias_correction: false
    folder: /home/.../era5ztquv/1979_2020/all_data

  - name: lmdz_35_present
    enable: true
    src: lmdz
    start: '2006-01-01'
    end: '2014-12-31'
    bias_correction: true
    folder: /home/.../LMDZ/r35
    bc_reference_folder: /home/.../LMDZ/r35

  - name: lmdz_250_present
    enable: true
    src: lmdz
    start: '2006-01-01'
    end: '2014-12-31'
    bias_correction: true
    folder: /home/.../LMDZ/r250
    bc_reference_folder: /home/.../LMDZ/r250
```

---

## 🚀 Lancement d'une Inférence

Pour exécuter une inférence avec votre configuration (`config.yaml`), soumettez simplement le script unifié :

```bash
sbatch run_inference.sh
```

Les journaux d'exécution seront automatiquement sauvegardés dans le dossier `logs/`.

---

## 📁 Organisation et Nettoyage (`tests/` & `logs/`)

Pour garder la racine claire et propre :
* **`tests/`** : Regroupe l'ensemble des anciennes configurations et scripts d'inférence spécifiques.
* **`logs/`** : Stocke tous les fichiers journaux (`out_*.log`) générés par SLURM.

