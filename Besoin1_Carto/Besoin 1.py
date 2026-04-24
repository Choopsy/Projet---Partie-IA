import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import plotly.express as px
from pyproj import Transformer

# 1. Chargement du jeu de données
#    - Le fichier export_IA.csv contient les arbres et leurs coordonnées en Lambert 93.
df = pd.read_csv('export_IA.csv')

# 2. Conversion géographique des coordonnées
#    - Transformer les coordonnées de Lambert 93 (EPSG:3949) vers WGS84 (EPSG:4326)
#    - WGS84 est utilisé par les cartes en ligne (latitude/longitude).
transformer = Transformer.from_crs("EPSG:3949", "EPSG:4326", always_xy=True)
df['lon'], df['lat'] = transformer.transform(df['x'].values, df['y'].values)

# 3. Préparation des données pour le clustering
#    - On ne conserve que les arbres avec hauteur et diamètre valides.
#    - On supprime les lignes sans coordonnées, hauteur ou diamètre.
features = ['haut_tot', 'tronc_diam']
df_clean = df.dropna(subset=features + ['lat', 'lon']).copy()
df_clean = df_clean[(df_clean['haut_tot'] > 0) & (df_clean['tronc_diam'] > 0)]

# 4. Standardisation des variables
#    - Les valeurs de hauteur et diamètre peuvent avoir des échelles très différentes.
#    - StandardScaler centre les données et les met à l'échelle pour KMeans.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# 5. Sauvegarde du scaler
#    - On garde le scaler pour appliquer la même transformation au moment de la prédiction.
pickle.dump(scaler, open('scaler_unique.pkl', 'wb'))

# 6. Comparaison des métriques de clustering pour K=2, 3 et 4
#    - Cela permet de choisir une bonne valeur de K selon la qualité du partitionnement.
print(f"{'K':<5} | {'Silhouette':<12} | {'Calinski-H':<12} | {'Davies-B':<10}")
print("-" * 45)

results = []
for k in [2, 3, 4]:
    # 6.1 Entraînement du modèle KMeans
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    # 6.2 Calcul des scores de qualité
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)

    print(f"{k:<5} | {sil:<12.4f} | {ch:<12.2f} | {db:<10.4f}")

    # 7. Construction d'un mapping lisible pour chaque cluster
    #    - On classe les clusters selon la moyenne de la hauteur.
    #    - Cela permet de donner des étiquettes compréhensibles.
    centers = df_clean.assign(cls=labels).groupby('cls')['haut_tot'].mean().sort_values()

    if k == 2:
        mapping = {centers.index[0]: "Petit", centers.index[1]: "Grand"}
    elif k == 3:
        mapping = {
            centers.index[0]: "Petit",
            centers.index[1]: "Moyen",
            centers.index[2]: "Grand"
        }
    else:
        mapping = {centers.index[i]: f"Taille_{i+1}" for i in range(k)}

    # 8. Sauvegarde des fichiers de modèle et de mapping
    #    - model_k{k}.pkl est le modèle KMeans entraîné.
    #    - mapping_k{k}.pkl associe chaque cluster à un nom humain.
    pickle.dump(model, open(f'model_k{k}.pkl', 'wb'))
    pickle.dump(mapping, open(f'mapping_k{k}.pkl', 'wb'))

    # 9. Génération de la carte interactive pour K=3 seulement
    #    - La carte est créée avec Plotly et enregistrée en HTML.
    #    - Ce fichier sera ensuite utilisé par predict_cluster.py.
    if k == 3:
        df_clean['cat'] = pd.Series(labels).map(mapping).values
        fig = px.scatter_mapbox(
            df_clean,
            lat="lat",
            lon="lon",
            color="cat",
            color_discrete_map={"Petit": "green", "Moyen": "orange", "Grand": "red"},
            hover_data={"lat": False, "lon": False, "haut_tot": True, "tronc_diam": True, "cat": True},
            mapbox_style="open-street-map",
            zoom=12,
            title="Visualisation du Patrimoine Arboré (K=3)"
        )
        fig.write_html("carte_interactive_besoin1.html")

print("-" * 45)
print("\nSystème prêt : Modèles K=2, 3, 4 sauvegardés et carte générée.")