# Prédiction de l'âge des arbres

---

## Partie 1 — Script d'estimation (`script.py`)

### Description

Ce script en ligne de commande permet d'estimer l'âge d'un arbre à partir de ses caractéristiques physiques et environnementales. Il utilise un modèle de Machine Learning pré-entraîné (Random Forest) ainsi qu'un scaler chargés depuis des fichiers `.pkl`.

### Prérequis

- Python 3.8+
- Bibliothèques : `numpy`, `joblib`

```bash
pip install numpy joblib
```

- Les fichiers suivants doivent être présents dans le répertoire du script :
  - `model_age_arbre.pkl` — le modèle entraîné
  - `scaler_age.pkl` — le scaler de normalisation

### Utilisation

```bash
python script.py
```

Au lancement, un menu s'affiche avec trois options :

```
----------------------------------------
  ESTIMATION DE L'AGE D'UN ARBRE
----------------------------------------
  1. Estimation simple
  2. Estimation complexe
  3. Quitter
----------------------------------------
```

#### Mode 1 — Estimation simple

L'utilisateur saisit uniquement les **3 mesures principales** de l'arbre :

| Variable | Description | Unité |
|---|---|---|
| `tronc_diam` | Diamètre du tronc | cm |
| `haut_tot` | Hauteur totale | m |
| `haut_tronc` | Hauteur du tronc | m |

Les variables supplémentaires sont automatiquement remplacées par leurs **valeurs moyennes** issues du dataset d'entraînement.

#### Mode 2 — Estimation complexe

En plus des 3 mesures de base, l'utilisateur peut renseigner les **variables qualitatives** suivantes (appuyer sur Entrée pour utiliser la valeur par défaut) :

| Variable | Valeurs possibles |
|---|---|
| `feuillage` | `conifère`, `feuillu`, `Inconnu` |
| `remarquable` | `Non`, `Oui` |
| `fk_pied` | `Bac de plantation`, `Bande de terre`, `fosse arbre`, `gazon`, `revetement non permeable`, `terre`, `toile tissée`, `végétation`, `NA` |
| `fk_port` | `tête de chat`, `têtard`, `semi libre`, `rideau`, `réduit`, `Libre`, `étêté`, `couronne`, `cépée`, `architecturé`, `Na`, … |
| `clc_quartier` | `Harly`, `Omissy`, `Quartier de l'Europe`, `Quartier du Centre-ville`, … |
| `clc_secteur` | *(valeur numérique ou Entrée pour défaut)* |

### Résultat

Le script affiche l'âge estimé de l'arbre en années :

```
----------------------------------------
AGE ESTIME : 47.3 ans
----------------------------------------
```

---

## Partie 2 — Entraînement du modèle (`ApprentissagePy.py`)

### Description

Ce script entraîne un modèle de **régression** pour prédire l'âge d'un arbre (`age_estim`) à partir d'un jeu de données tabulaire. Il compare plusieurs algorithmes, analyse l'apport de chaque variable, optimise les hyperparamètres, puis exporte le modèle final.

### Prérequis

- Python 3.8+
- Bibliothèques : `pandas`, `numpy`, `scikit-learn`, `joblib`

```bash
pip install pandas numpy scikit-learn joblib
```

- Le fichier de données `DataForIA.csv` (séparateur `;`) doit être accessible. Modifier le chemin en tête de script :

```python
data = pd.read_csv(r"chemin/vers/DataForIA.csv", sep=";")
```

### Variables utilisées

| Variable | Type | Rôle |
|---|---|---|
| `tronc_diam` | Numérique | Feature |
| `haut_tot` | Numérique | Feature |
| `haut_tronc` | Numérique | Feature |
| `feuillage` | Catégorielle | Feature |
| `remarquable` | Catégorielle | Feature |
| `fk_pied` | Catégorielle | Feature |
| `fk_port` | Catégorielle | Feature |
| `clc_quartier` | Catégorielle | Feature |
| `clc_secteur` | Catégorielle | Feature |
| `age_estim` | Numérique | **Cible (y)** |

### Étapes du pipeline

1. **Chargement et nettoyage** — suppression des lignes avec des valeurs manquantes (`dropna`).
2. **Encodage** — les variables qualitatives sont encodées avec `LabelEncoder`.
3. **Découpage** — 80% des données pour l'entraînement, 20% pour le test (`random_state=42`).
4. **Normalisation** — application d'un `StandardScaler` (ajusté sur le train uniquement).
5. **Comparaison de modèles** — 4 algorithmes sont évalués sur MAE, RMSE et R² :
   - `LinearRegression`
   - `DecisionTreeRegressor`
   - `RandomForestRegressor`
   - `GradientBoostingRegressor`
6. **Analyse par nombre de features** — évaluation du R² en ajoutant les variables une par une pour identifier les plus contributives.
7. **Optimisation (GridSearchCV)** — recherche des meilleurs hyperparamètres du `RandomForestRegressor` par validation croisée à 5 folds :
   - `n_estimators` : [20, 100, 200]
   - `max_depth` : [None, 10, 20]
   - `min_samples_leaf` : [1, 2, 4]
8. **Export** — sauvegarde du modèle optimisé et du scaler :
   - `model_age_arbre.pkl`
   - `scaler_age.pkl`

### Lancement

```bash
python ApprentissagePy.py
```

### Fichiers générés

| Fichier | Description |
|---|---|
| `model_age_arbre.pkl` | Modèle Random Forest optimisé |
| `scaler_age.pkl` | Scaler de normalisation (indispensable pour le script) |

> **Important :** Ces deux fichiers doivent être placés dans le même répertoire que `script.py` pour que l'estimation fonctionne correctement.