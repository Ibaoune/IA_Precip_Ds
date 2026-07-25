# Tests des Prédicteurs (Variables d'Entrée)

Ce dossier regroupe les expériences visant à évaluer l'impact et la sensibilité du modèle U-Net face à l'ajout de variables géographiques et climatiques explicites (prédicteurs) :

- **config_unet_mslp.yaml** : Ajout de la pression au niveau de la mer (Mean Sea Level Pressure).
- **config_unet_topo.yaml** : Ajout de la topographie (Altitude statique) pour mieux contraindre les précipitations orographiques.
- **config_unet_topo_mslp.yaml** : Utilisation combinée de la topographie et de la pression au niveau de la mer.

Ces expériences aident à déterminer si l'ajout d'informations physiques complémentaires aux variables dynamiques classiques (humidité, température, vents) améliore significativement la descente d'échelle (downscaling).
