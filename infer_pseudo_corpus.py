import pandas as pd
import numpy as np
import torch
import pickle
import scipy.spatial.distance as dist
from run_verification import SiameseTabularNet


def compute_asterius_profile(model, scaler, train_csv_path):

    df_train = pd.read_csv(train_csv_path)
    feature_cols = df_train.columns.drop(['Auteur', 'Titre'])
    df_asterius = df_train[df_train['Auteur'].str.lower() == 'asterius'].copy()

    if df_asterius.empty:
        raise ValueError("Kritischer Fehler: Keine Asterius-Texte in den Trainingsdaten gefunden.")

    X_ast_scaled = scaler.transform(df_asterius[feature_cols])

    with torch.no_grad():
        ast_embeddings = model(torch.FloatTensor(X_ast_scaled)).numpy()
    asterius_centroid = np.mean(ast_embeddings, axis=0)
    distances_to_centroid = [dist.euclidean(emb, asterius_centroid) for emb in ast_embeddings]
    dynamic_threshold = np.mean(distances_to_centroid) + 2 * np.std(distances_to_centroid)

    return asterius_centroid, dynamic_threshold


def run_inference(train_csv, pseudo_csv, model_path, scaler_path, output_csv):
    print("--- Schritt 1: Lade Modelle und Metadaten ---")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df_train = pd.read_csv(train_csv)
    feature_cols = df_train.columns.drop(['Auteur', 'Titre'])

    model = SiameseTabularNet(input_size=len(feature_cols))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print("\n--- Schritt 2: Generiere Asterius-Referenzprofil (Zentroid) ---")
    asterius_centroid, dynamic_threshold = compute_asterius_profile(model, scaler, train_csv)
    print(f"[INFO] Dynamischer Intra-Autor-Schwellenwert (Threshold): {dynamic_threshold:.4f}")

    print("\n--- Schritt 3: Evaluiere anonymes Pseudo-Chrysostomos-Korpus ---")
    df_pseudo = pd.read_csv(pseudo_csv)
    pseudo_labels = df_pseudo['Auteur'] if 'Auteur' in df_pseudo.columns else pd.Series(["Unbekannt"] * len(df_pseudo))
    titles = df_pseudo['Titre']

    pseudo_features = df_pseudo.reindex(columns=feature_cols, fill_value=0)
    X_pseudo_scaled = scaler.transform(pseudo_features)

    with torch.no_grad():
        pseudo_embeddings = model(torch.FloatTensor(X_pseudo_scaled)).numpy()

    print("\n--- Schritt 4: Distanzberechnung und automatisierte Klassifizierung ---")
    results = []
    for i, emb in enumerate(pseudo_embeddings):
        distance = dist.euclidean(emb, asterius_centroid)

        if distance <= dynamic_threshold:
            attribution = "Core-Asterius (Sichere Zuweisung)"
        elif distance <= (dynamic_threshold * 1.25):
            attribution = "Grauzone (Theologisch verwandt / Kompilation)"
        else:
            attribution = "Abgewiesen (Fremdautor)"

        results.append({
            'ComparatorClass': 'Asterius',
            'Pseudo_Text_Titre': titles.iloc[i],
            'Distance_to_Centroid': distance,
            'Original_Klasse': pseudo_labels.iloc[i],
            'Dynamic_Threshold': dynamic_threshold,  # Threshold für die Aggregation exportieren
            'Classification': attribution
        })

    df_results = pd.DataFrame(results).sort_values(by='Distance_to_Centroid')
    df_results.to_csv(output_csv, index=False)

    print(f"[OK] Inferenz abgeschlossen. Ergebnisse gespeichert unter: {output_csv}\n")
    print("Top 5 Kandidaten für eine Autorschaft durch Asterius:")
    print(df_results[['Pseudo_Text_Titre', 'Distance_to_Centroid', 'Classification']].head(5).to_string(index=False))


if __name__ == "__main__":
    TRAIN_CSV = "train_features.csv"
    PSEUDO_CSV = "inference_features.csv"
    MODEL_PTH = "siamese_asterius.pth"
    SCALER_PKL = "scaler.pkl"
    OUTPUT_CSV = "asterius_inference_results.csv"

    run_inference(TRAIN_CSV, PSEUDO_CSV, MODEL_PTH, SCALER_PKL, OUTPUT_CSV)