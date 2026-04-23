import pandas as pd
import numpy as np
import plotly.express as px
from pyproj import Transformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib

# --- ÉTAPE 1 : Préparation des données ---
df = pd.read_csv('export_IA.csv')

# Conversion des coordonnées GPS (Code EPSG 3949 trouvé dans ton script R)
transformer = Transformer.from_crs("EPSG:3949", "EPSG:4326", always_xy=True)
df['lon'], df['lat'] = transformer.transform(df['x'].values, df['y'].values)

# Sélection des variables d'intérêt : hauteur et diamètre
features = ['haut_tot', 'tronc_diam']
df_clean = df.dropna(subset=features + ['lat', 'lon']).copy()

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# --- ÉTAPE 2 : Apprentissage non-supervisé ---
n_clusters = 3  # Choix de l'utilisateur (ex: 3 pour petit, moyen, grand)
model_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df_clean['cluster'] = model_kmeans.fit_predict(X_scaled)

# --- ÉTAPE 3 : Métriques (Indispensables pour ton rapport) ---
sil = silhouette_score(X_scaled, df_clean['cluster'])
cali = calinski_harabasz_score(X_scaled, df_clean['cluster'])
print(f"Silhouette Score: {sil:.3f}")
print(f"Calinski-Harabasz Index: {cali:.3f}")

# SAUVEGARDE (Pour l'étape 5)
joblib.dump(model_kmeans, 'model_clustering.pkl')
joblib.dump(scaler, 'scaler_clustering.pkl')

# --- ÉTAPE 4 : Visualisation sur carte ---
fig = px.scatter_mapbox(df_clean, lat="lat", lon="lon", color=df_clean['cluster'].astype(str),
                        hover_data=['haut_tot', 'tronc_diam'],
                        zoom=13, mapbox_style="open-street-map",
                        title=f"Clustering des arbres par taille ({n_clusters} catégories)")
fig.show()