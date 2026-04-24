import pickle
import numpy as np
import os
import webbrowser  # Pour ouvrir la carte automatiquement

def main():
    print("====================================================")
    print("   SYSTÈME DE PRÉDICTION DE TAILLE (SAINT-QUENTIN)")
    print("====================================================\n")

    # 1. Sélection du modèle
    k_choice = input("Choisissez le nombre de catégories (2, 3 ou 4) : ")
    
    try:
        model = pickle.load(open(f'model_k{k_choice}.pkl', 'rb'))
        mapping = pickle.load(open(f'mapping_k{k_choice}.pkl', 'rb'))
        scaler = pickle.load(open('scaler_unique.pkl', 'rb'))

        # 2. Saisie des caractéristiques
        h = float(input("Entrez la hauteur (m) : ").replace(',', '.'))
        d = float(input("Entrez le diamètre (cm) : ").replace(',', '.'))

        # 3. Prédiction
        data_scaled = scaler.transform(np.array([[h, d]]))
        cluster_id = model.predict(data_scaled)[0]
        
        print(f"\n>>> RÉSULTAT : Catégorie **{mapping[cluster_id]}**")
        
        
        # --- AJOUT : LANCEMENT DE LA CARTE ---
        print("\nOuverture de la carte interactive dans votre navigateur...")
        # On définit le chemin de la carte générée lors de l'entraînement
        path = os.path.abspath("carte_interactive_besoin1.html")
        webbrowser.open(f"file://{path}")

    except FileNotFoundError:
        print(f"Erreur : Le modèle pour {k_choice} clusters n'existe pas.")

if __name__ == "__main__":
    main()
    