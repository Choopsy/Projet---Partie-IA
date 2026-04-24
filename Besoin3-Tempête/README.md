# Besoin Client 3 — Système d'alerte pour les tempêtes

## Prérequis

Installer les packages nécessaires :
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Utilisation du script

```bash
python predict_tempete.py 
```

**Colonnes requises dans le CSV d'entrée :**
```
haut_tot, haut_tronc, tronc_diam, age_estim, fk_stadedev,
fk_port, fk_pied, fk_situation, fk_revetement, clc_quartier,
feuillage, remarquable, fk_nomtech, clc_nbr_diag
```

---

## Exemple de sortie terminal

```
Chargement du fichier : export_IA.csv
   9915 arbres chargés.
Modèle et préprocesseurs chargés avec succès.

Prétraitement des données...
Prédiction en cours...

Prédictions exportées dans : resultats.csv

Résumé des prédictions :
   Stable         (EN PLACE)                     : 8870 arbre(s)
   Danger élevé   (Non essouché — déraciné)      : 65 arbre(s)
   Risque modéré  (Essouché — a été déraciné)    : 189 arbre(s)
   Risque faible  (Abattu + Supprimé + Remplacé) : 791 arbre(s)
```

---

## Description des fichiers

| Fichier | Description |
|---|---|
| `predict_tempete.py` | Script principal de prédiction |
| `Besoin_Client_3.ipynb` | Notebook commenté — démarche expérimentale complète |
| `export_IA.csv` | Données d'entrée |
| `./modeles/model_rf.pkl` | Modèle Random Forest entraîné |
| `./modeles/encoders.pkl` | Encodeurs des variables catégorielles |
| `./modeles/scaler.pkl` | Normaliseur des variables numériques |