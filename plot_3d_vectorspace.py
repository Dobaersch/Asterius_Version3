import pandas as pd
import torch
import pickle
import plotly.express as px
from sklearn.manifold import TSNE
import os

# Zwingender Import der Architektur
from run_verification import SiameseTabularNet

def generate_interactive_3d_space(train_csv, pseudo_csv, model_path, scaler_path):
    print("1. Lade Persistenz-Artefakte und Daten...")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Modell oder Scaler fehlen. Bitte zuerst Training durchführen.")
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    df_train = pd.read_csv(train_csv)
    df_pseudo = pd.read_csv(pseudo_csv)
    
    feature_cols = df_train.columns.drop(['Auteur', 'Titre'])
    
    # 2. Modell initialisieren
    model = SiameseTabularNet(input_size=len(feature_cols))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # 3. Gemeinsamen Datensatz für t-SNE vorbereiten
    # Wir markieren die Herkunft, um sie im Plot visuell zu trennen
    df_train['Korpus'] = 'Referenz (Gesichert)'
    
    # Pseudo-Korpus anpassen (falls 'Auteur' fehlt, Titel als Label nutzen)
    if 'Auteur' not in df_pseudo.columns:
        df_pseudo['Auteur'] = 'Unbekannt'
    df_pseudo['Korpus'] = 'Pseudo-Chrysostomos (Inferenz)'
    
    df_combined = pd.concat([df_train, df_pseudo], ignore_index=True)
    
    # Skalierung zwingend auf dem gesamten Set anwenden
    X_scaled = scaler.transform(df_combined[feature_cols])
    
    # 4. 64-dimensionale Embeddings generieren
    print("2. Generiere hochdimensionale Embeddings...")
    with torch.no_grad():
        embeddings_64d = model(torch.FloatTensor(X_scaled)).numpy()
        
    # 5. Dimensionsreduktion auf exakt 3 Dimensionen
    print("3. Führe 3D t-SNE Reduktion durch (dies kann einen Moment dauern)...")
    tsne_3d = TSNE(n_components=3, perplexity=min(30, len(df_combined)-1), random_state=42)
    embeddings_3d = tsne_3d.fit_transform(embeddings_64d)
    
    # 6. Plot-Dataframe konstruieren
    plot_df = pd.DataFrame({
        'X': embeddings_3d[:, 0],
        'Y': embeddings_3d[:, 1],
        'Z': embeddings_3d[:, 2],
        'Autor': df_combined['Auteur'],
        'Titel': df_combined['Titre'],
        'Korpus': df_combined['Korpus']
    })
    
    # 7. Interaktiven Plotly-Graph generieren
    print("4. Erstelle interaktive HTML-Datei...")
    
    # Farbzuweisung für Konsistenz
    color_discrete_map = {
        'Asterius': '#d62728',       # Rot
        'Chrysostomos': '#2ca02c',   # Grün
        'Severian': '#1f77b4',       # Blau
        'Unbekannt': '#ff7f0e'       # Orange für Pseudo-Texte
    }
    
    fig = px.scatter_3d(
        plot_df, 
        x='X', y='Y', z='Z',
        color='Autor',
        symbol='Korpus',
        hover_name='Titel', # Zeigt den Predigttitel beim Mouse-Over
        hover_data={'Autor': True, 'Korpus': False, 'X': False, 'Y': False, 'Z': False},
        color_discrete_map=color_discrete_map,
        title='Interaktiver Vektorraum: Asterius-Attribution',
        opacity=0.8
    )
    
    # Optisches Layout für Präsentationen anpassen
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(
        scene=dict(
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            zaxis_title='t-SNE Dimension 3',
            bgcolor='whitesmoke'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    output_html = "interaktiver_vektorraum.html"
    fig.write_html(output_html)
    print(f"[ERFOLG] 3D-Visualisierung gespeichert als '{output_html}'. Öffne diese Datei in deinem Browser.")

if __name__ == "__main__":
    TRAIN_CSV = "train_features.csv"
    PSEUDO_CSV = "inference_features.csv"
    MODEL_PTH = "siamese_asterius.pth"
    SCALER_PKL = "scaler.pkl"
    
    generate_interactive_3d_space(TRAIN_CSV, PSEUDO_CSV, MODEL_PTH, SCALER_PKL)