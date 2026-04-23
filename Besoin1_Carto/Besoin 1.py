import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
import plotly.express as px
from pyproj import Transformer

# 1. Chargement et conversion (Lambert 93 -> WGS84) 
df = pd.read_csv('export_IA.csv')
transformer = Transformer.from_crs("EPSG:3949", "EPSG:4326", always_xy=True)
df['lon'], df['lat'] = transformer.transform(df['x'].values, df['y'].values)

# 2. Préparation des données (Hauteur et Diamètre) 
features = ['haut_tot', 'tronc_diam',]
df_clean = df.dropna(subset=features + ['lat', 'lon']).copy()
df_clean = df_clean[(df_clean['haut_tot'] > 0) & (df_clean['tronc_diam'] > 0)]

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(df_clean[features])
pickle.dump(scaler, open('scaler_unique.pkl', 'wb')) 

# --- ÉTAPE 1 : COMPARATIF DES MÉTRIQUES (K=2, 3, 4) ---
print(f"{'K':<5} | {'Silhouette':<12} | {'Calinski-H':<12} | {'Davies-B':<10}")
print("-" * 45)

results = []
for k in [2, 3, 4]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10) 
    labels = model.fit_predict(X_scaled)
    
    sil = silhouette_score(X_scaled, labels) 
    ch = calinski_harabasz_score(X_scaled, labels) 
    db = davies_bouldin_score(X_scaled, labels) 
    
    print(f"{k:<5} | {sil:<12.4f} | {ch:<12.2f} | {db:<10.4f}")
    
    # --- ÉTAPE 2 : SAUVEGARDE DES MODÈLES ---
    # Mapping des noms selon la hauteur moyenne pour le script final
    centers = df_clean.assign(cls=labels).groupby('cls')['haut_tot'].mean().sort_values()
    
    if k == 2: 
        mapping = {centers.index[0]: "Petit", centers.index[1]: "Grand"}
    elif k == 3: 
        mapping = {centers.index[0]: "Petit", centers.index[1]: "Moyen", centers.index[2]: "Grand"}
    else: 
        mapping = {centers.index[i]: f"Taille_{i+1}" for i in range(k)}

    # Sauvegarde 
    pickle.dump(model, open(f'model_k{k}.pkl', 'wb'))
    pickle.dump(mapping, open(f'mapping_k{k}.pkl', 'wb'))
    
    # On génère la carte interactive pour le cas K=3 par défaut 
    if k == 3:
        df_clean['cat'] = pd.Series(labels).map(mapping).values
        fig = px.scatter_mapbox(df_clean, lat="lat", lon="lon", color="cat",
                                color_discrete_map={"Petit": "green", "Moyen": "orange", "Grand": "red"},
                                hover_data={"lat":False, "lon":False, "haut_tot":True, "tronc_diam":True, "cat":True},
                                mapbox_style="open-street-map", zoom=12, 
                                title="Visualisation du Patrimoine Arboré (K=3)")
        fig.write_html("carte_interactive_besoin1.html")

print("-" * 45)
print("\nSystème prêt : Modèles K=2, 3, 4 sauvegardés et carte générée.")