import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(model, scaler, csv_path):
    """
    Extrahiert die Einbettungen des trainierten Netzwerks und 
    visualisiert sie mittels PCA und t-SNE.
    """
    # 1. Daten laden und vorbereiten
    df = pd.read_csv(csv_path)
    labels = df['Auteur'].values
    titles = df['Titre'].values
    feature_cols = df.columns.drop(['Auteur', 'Titre'])
    
    # Exakt gleiche Skalierung wie im Training anwenden
    X_scaled = scaler.transform(df[feature_cols])
    
    # 2. Modell in Evaluierungsmodus versetzen und Embeddings generieren
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        embeddings = model(X_tensor).numpy() # 64-dimensionale Vektoren
        
    # 3. Dimensionsreduktion berechnen
    print("Berechne PCA...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings)
    
    print("Berechne t-SNE...")
    # Perplexity an die Größe des Datensatzes anpassen (oft zwischen 5 und 50)
    perplexity = min(30, len(df) - 1) 
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    # 4. DataFrame für Visualisierung erstellen
    vis_df = pd.DataFrame({
        'Auteur': labels,
        'Titre': titles,
        'PCA1': pca_results[:, 0], 'PCA2': pca_results[:, 1],
        'TSNE1': tsne_results[:, 0], 'TSNE2': tsne_results[:, 1]
    })
    
    # 5. Plotting (Seaborn für akademisch saubere Graphen)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_style("whitegrid")
    
    # Farbpalette definieren (Asterius hervorheben)
    palette = sns.color_palette("husl", len(set(labels)))
    
    # PCA Plot
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='Auteur', style='Auteur',
        palette=palette, data=vis_df, ax=axes[0], s=60, alpha=0.8
    )
    axes[0].set_title('PCA der Autor-Embeddings (Globale Struktur)', fontsize=14)
    
    # t-SNE Plot
    sns.scatterplot(
        x='TSNE1', y='TSNE2', hue='Auteur', style='Auteur',
        palette=palette, data=vis_df, ax=axes[1], s=60, alpha=0.8
    )
    axes[1].set_title('t-SNE der Autor-Embeddings (Lokale Cluster)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('embedding_validation.png', dpi=300)
    print("Visualisierung gespeichert unter 'embedding_validation.png'.")
    plt.show()

# Aufruf (benötigt das trainierte Modell und den Scaler aus run_verification.py)
# visualize_embeddings(trained_model, fitted_scaler, 'pfad/zu/deinen/trainingsdaten.csv')