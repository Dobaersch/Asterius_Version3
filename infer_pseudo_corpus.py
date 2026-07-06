import pandas as pd
import numpy as np
import torch
import pickle
import scipy.spatial.distance as dist

# =====================================================================
# 1. ZWINGENDER ARCHITEKTUR-IMPORT (Vermeidung von Code-Redundanz)
# =====================================================================
# Das Netzwerk muss exakt so aufgebaut sein wie im Training.
# Durch den Import wird sichergestellt, dass Änderungen im Trainingsskript
# automatisch auch in der Inferenz übernommen werden.
from run_verification import SiameseTabularNet


# =====================================================================
# 2. PIPELINE-FUNKTIONEN
# =====================================================================
def compute_asterius_profile(model, scaler, train_csv_path):
    """
    Berechnet das Asterius-Zentroid (den idealen Stil-Mittelpunkt) und den
    Intra-Autor-Schwellenwert basierend auf den gesicherten Trainingsdaten.
    """
    df_train = pd.read_csv(train_csv_path)
    feature_cols = df_train.columns.drop(['Auteur', 'Titre'])

    # Nur Asterius-Texte filtern (case-insensitive, um Fehler zu vermeiden)
    df_asterius = df_train[df_train['Auteur'].str.lower() == 'asterius'].copy()

    if df_asterius.empty:
        raise ValueError("Kritischer Fehler: Keine Asterius-Texte in den Trainingsdaten gefunden.")

    X_ast_scaled = scaler.transform(df_asterius[feature_cols])

    with torch.no_grad():
        ast_embeddings = model(torch.FloatTensor(X_ast_scaled)).numpy()

    # 1. Berechne das Zentroid (den "idealen" Asterius-Stil)
    asterius_centroid = np.mean(ast_embeddings, axis=0)

    # 2. Berechne die Intra-Autor-Distanzen (Schwankungsbreite der gesicherten Texte)
    distances_to_centroid = [dist.euclidean(emb, asterius_centroid) for emb in ast_embeddings]

    # 3. Dynamischer Schwellenwert (Maximale Abweichung, die Asterius sich selbst erlaubt)
    dynamic_threshold = np.max(distances_to_centroid)

    return asterius_centroid, dynamic_threshold


def run_inference(train_csv, pseudo_csv, model_path, scaler_path, output_csv):
    print("--- Schritt 1: Lade Modelle und Metadaten ---")

    # Korrekter Import über pickle, passend zum Export in run_verification.py
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"[OK] Scaler aus '{scaler_path}' geladen.")

    df_train = pd.read_csv(train_csv)
    feature_cols = df_train.columns.drop(['Auteur', 'Titre'])

    # Architektur aufbauen und evaluieren
    model = SiameseTabularNet(input_size=len(feature_cols))
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"[OK] Gewichte geladen. Netzwerkarchitektur erfolgreich importiert.")

    print("\n--- Schritt 2: Generiere Asterius-Referenzprofil (Zentroid) ---")
    asterius_centroid, dynamic_threshold = compute_asterius_profile(model, scaler, train_csv)
    print(f"[INFO] Asterius-Zentroid erfolgreich im 64-dimensionalen Raum lokalisiert.")
    print(f"[INFO] Dynamischer Intra-Autor-Schwellenwert (Threshold): {dynamic_threshold:.4f}")

    print("\n--- Schritt 3: Evaluiere anonymes Pseudo-Chrysostomos-Korpus ---")
    df_pseudo = pd.read_csv(pseudo_csv)

    # Dynamische Extraktion der Metadaten
    if 'Auteur' in df_pseudo.columns:
        pseudo_labels = df_pseudo['Auteur']
    else:
        pseudo_labels = pd.Series(["Unbekannt"] * len(df_pseudo))

    titles = df_pseudo['Titre']

    pseudo_features = df_pseudo.reindex(columns=feature_cols, fill_value=0)

    # Skalieren streng nach dem Trainings-Maßstab (Verhinderung von Data Leakage)
    X_pseudo_scaled = scaler.transform(pseudo_features)

    with torch.no_grad():
        pseudo_embeddings = model(torch.FloatTensor(X_pseudo_scaled)).numpy()

    print("\n--- Schritt 4: Distanzberechnung und automatisierte Klassifizierung ---")
    results = []
    for i, emb in enumerate(pseudo_embeddings):
        distance = dist.euclidean(emb, asterius_centroid)

        # Automatisierte Autorschafts-Zuweisung auf Basis der errechneten Schwankungsbreite
        if distance <= dynamic_threshold:
            attribution = "Core-Asterius (Sichere Zuweisung)"
        elif distance <= (dynamic_threshold * 1.25):  # 25% Toleranz für Genre-Rauschen
            attribution = "Grauzone (Theologisch verwandt / Kompilation)"
        else:
            attribution = "Abgewiesen (Fremdautor)"


        results.append({
            'ComparatorClass': 'Asterius',
            'ComparedLabel': titles.iloc[i],
            'Distance': distance,
            'Original_Klasse': pseudo_labels.iloc[i],
            'Threshold_Baseline': dynamic_threshold,
            'Classification': attribution
        })

    # 5. Ergebnisse als DataFrame aufbereiten und speichern
    df_results = pd.DataFrame(results).sort_values(by='Distance')

    # Hinweis: Wenn output_csv in main() "asterius_inference_results.csv" heißt,
    # muss aggregate_results.py diesen Dateinamen beim Laden (pd.read_csv) nutzen!
    df_results.to_csv(output_csv, index=False)

    print(f"[OK] Inferenz abgeschlossen. Ergebnisse gespeichert unter: {output_csv}\n")
    print("Top 5 Kandidaten für eine Autorschaft durch Asterius:")
    # Ausgabe für die Konsole verschlanken
    print(df_results[['Pseudo_Text_Titre', 'Distance_to_Centroid', 'Classification']].head(5).to_string(index=False))


# =====================================================================
# 3. AUSFÜHRUNGSBLOCK
# =====================================================================
if __name__ == "__main__":
    # Bitte überprüfe, ob der Dateiname deiner Pseudo-CSV exakt mit diesem übereinstimmt
    TRAIN_CSV = "train_features.csv"
    PSEUDO_CSV = "inference_features.csv"
    MODEL_PTH = "siamese_asterius.pth"
    SCALER_PKL = "scaler.pkl"
    OUTPUT_CSV = "asterius_inference_results.csv"

    run_inference(TRAIN_CSV, PSEUDO_CSV, MODEL_PTH, SCALER_PKL, OUTPUT_CSV)