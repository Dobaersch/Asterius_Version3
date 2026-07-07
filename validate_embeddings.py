import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import os

# WICHTIG: Import der Mikro-Architektur
from run_verification import SiameseTabularNet


def visualize_embeddings(model, scaler, pca, csv_path):
    """
    Extrahiert die Einbettungen des trainierten Netzwerks und
    visualisiert sie mittels PCA und t-SNE im 2D-Raum.
    """
    # 1. Daten laden und vorbereiten
    df = pd.read_csv(csv_path)
    labels = df['Auteur'].values
    titles = df['Titre'].values
    feature_cols = df.columns.drop(['Auteur', 'Titre'])

    # 2. Exakt gleiche Skalierung UND PCA wie im Training anwenden!
    X_raw = df[feature_cols].astype(float)
    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)  # Diese Zeile fehlte bisher

    # 3. Modell in Evaluierungsmodus versetzen und Embeddings generieren
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_pca)
        # Das Modell gibt nun unsere extrem dichten 8-dimensionalen Vektoren zurück
        embeddings = model(X_tensor).numpy()

        # 4. Dimensionsreduktion für den Plot berechnen
    print("Berechne PCA für 2D Plot...")
    plot_pca = PCA(n_components=2)
    pca_results = plot_pca.fit_transform(embeddings)

    print("Berechne t-SNE für 2D Plot...")
    # Perplexity dynamisch an kleine Datensätze anpassen, um Abstürze zu vermeiden
    perplexity = min(30, len(embeddings) - 1) if len(embeddings) > 1 else 1
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    # 5. Visualisierung
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    def plot_scatter(ax, data, title):
        sns.scatterplot(
            x=data[:, 0], y=data[:, 1],
            hue=labels,
            palette="deep",
            ax=ax,
            s=100,
            alpha=0.8
        )
        ax.set_title(title)

        # Titel nur für Asterius-Texte einblenden, um den Plot nicht zu überladen
        for i, txt in enumerate(titles):
            if labels[i].lower() == 'asterius':
                ax.annotate(txt[:15] + "...", (data[i, 0], data[i, 1]), fontsize=8, alpha=0.7)

    plot_scatter(ax1, pca_results, 'PCA der Siamese Embeddings')
    plot_scatter(ax2, tsne_results, 't-SNE der Siamese Embeddings')

    plt.tight_layout()
    plt.savefig("embedding_visualisierung.png", dpi=300)
    print("[OK] Visualisierung gespeichert unter 'embedding_visualisierung.png'")
    plt.show()


if __name__ == "__main__":
    csv_pfad = "train_features.csv"
    modell_pfad = "siamese_asterius.pth"
    scaler_pfad = "scaler.pkl"
    pca_pfad = "pca_model.pkl"  # Der neue PCA-Pfad

    # 1. Lade den Z-Standardisierungs-Scaler
    print("Lade persistierten Scaler...")
    with open(scaler_pfad, "rb") as f:
        geladener_scaler = pickle.load(f)

    # 2. Lade die PCA-Matrix
    print("Lade persistierte PCA-Matrix...")
    with open(pca_pfad, "rb") as f:
        geladene_pca = pickle.load(f)

    # 3. Lade das trainierte Modell
    print("Lade Modellgewichte...")

    # Das Netzwerk darf nicht mehr mit der Länge der rohen Spalten (241) aufgebaut werden,
    # sondern zwingend mit der Anzahl der PCA-Komponenten (15).
    input_size = geladene_pca.n_components_

    trainiertes_modell = SiameseTabularNet(input_size)
    trainiertes_modell.load_state_dict(torch.load(modell_pfad, weights_only=True))

    visualize_embeddings(trainiertes_modell, geladener_scaler, geladene_pca, csv_pfad)