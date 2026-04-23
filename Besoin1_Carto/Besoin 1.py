import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import plotly.express as px
from pyproj import Transformer

# 1. Chargement et conversion des coordonnées (Lambert 93 -> WGS84)
# S'inspire de la préparation de données du module Big Data
df = pd.read_csv('export_IA.csv')
transformer = Transformer.from_crs("EPSG:3949", "EPSG:4326", always_xy=True)
df['lon'], df['lat'] = transformer.transform(df['x'].values, df['y'].values)

# 2. Préparation des données : Hauteur et Diamètre 
features = ['haut_tot', 'tronc_diam']
df_clean = df.dropna(subset=features + ['lat', 'lon']).copy()
# Filtrage des données aberrantes pour améliorer les scores
df_clean = df_clean[(df_clean['haut_tot'] > 0) & (df_clean['tronc_diam'] > 0)]

# Normalisation : Crucial quand on utilise deux unités différentes (m et cm) [cite: 117]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# 3. Clustering K-Means (3 catégories : Petit, Moyen, Grand) [cite: 102, 120]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df_clean['cluster_id'] = labels

# 4. ÉVALUATION (Métriques demandées par le client) [cite: 65, 125]
sil = silhouette_score(X_scaled, labels)
ch = calinski_harabasz_score(X_scaled, labels)
db = davies_bouldin_score(X_scaled, labels)

print(f"--- Évaluation de la qualité (Hauteur + Diamètre) ---")
print(f"1. Silhouette Coefficient (Cible ~ 1) : {sil:.4f}")
print(f"2. Calinski-Harabasz Index (Plus haut est mieux) : {ch:.2f}")
print(f"3. Davies-Bouldin Index (Plus bas est mieux) : {db:.4f}\n")

# 5. Nommage des catégories par volume (Hauteur * Diamètre moyen)
# On s'assure que le cluster 0 est toujours 'Petit', etc.
df_clean['volume_approx'] = df_clean['haut_tot'] * df_clean['tronc_diam']
centers = df_clean.groupby('cluster_id')['volume_approx'].mean().sort_values()
mapping = {centers.index[0]: "Petit", centers.index[1]: "Moyen", centers.index[2]: "Grand"}
df_clean['categorie_taille'] = df_clean['cluster_id'].map(mapping)

# 6. Sauvegarde des modèles pour le script de prédiction [cite: 132]
pickle.dump(kmeans, open('model_besoin1.pkl', 'wb'))
pickle.dump(scaler, open('scaler_besoin1.pkl', 'wb'))
pickle.dump(mapping, open('mapping_clusters.pkl', 'wb'))

# 7. Carte Interactive [cite: 107, 127]
fig = px.scatter_mapbox(
    df_clean, lat="lat", lon="lon", color="categorie_taille",
    color_discrete_map={"Petit": "green", "Moyen": "orange", "Grand": "red"},
    hover_data={"lat": False, "lon": False, "haut_tot": True, "tronc_diam": True, "categorie_taille": True},
    zoom=12, mapbox_style="open-street-map",
    title="Besoin 1 : Classification par Taille (Hauteur & Diamètre)"
)
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
fig.write_html("carte_interactive_besoin1.html")
print("Carte et modèles générés avec succès.")