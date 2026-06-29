import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from scipy.spatial import distance


# =====================================================================
# 1. ARCHITEKTUR-DEFINITION
# =====================================================================
class SiameseTabularNet(nn.Module):
    def __init__(self, input_size):
        super(SiameseTabularNet, self).__init__()
        # Identische Struktur wie im Trainingsskript (run_verification.py)
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64)  # 64-dimensionaler Einbettungsraum
        )

    def forward(self, x):
        return self.fc(x)


# =====================================================================
# 2. PIPELINE-FUNKTIONEN
# =====================================================================
def compute_asterius_profile(model, scaler, train_csv_path):
    """
    Berechnet das Asterius-Zentroid und den maximalen internen Distanz-Schwellenwert.
    """
    df_train = pd.read_csv(train_csv_path)

    # Filtere ausschließlich verifizierte Asterius-Texte aus dem Trainingsset
    asterius_df = df_train[df_train['Auteur'] == 'asterius']
    if asterius_df.empty:
        # Fallback für unterschiedliche Schreibweisen
        asterius_df = df_train[df_train['Auteur'].str.lower() == 'asterius']

    if asterius_df.empty:
        raise ValueError(f"Keine echten Asterius-Texte in {train_csv_path} unter der Klasse 'Auteur' gefunden.")

    feature_cols = df_train.columns.drop(['Auteur', 'Titre'])

    # Skalieren mit dem im Training gefitteten Scaler
    X_scaled = scaler.transform(asterius_df[feature_cols])

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        embeddings = model(X_tensor).numpy()

    # Zentroid (Durchschnittsvektor aller Asterius-Samples) berechnen
    centroid = np.mean(embeddings, axis=0)

    # Schwellenwert berechnen (Maximale euklidische Distanz eines echten Textes zum Zentroid)
    distances = [distance.euclidean(centroid, emb) for emb in embeddings]
    max_intra_distance = np.max(distances)

    # 15% statistische Toleranz (Margin) hinzufügen, um stilistische Varianz zu erlauben
    threshold = max_intra_distance * 1.15

    print(f"-> Asterius-Zentroid erfolgreich berechnet.")
    print(f"-> Empirischer Intra-Autor-Schwellenwert: {threshold:.4f}")
    return centroid, threshold, feature_cols


def evaluate_pseudo_corpus(model, scaler, centroid, threshold, feature_cols, pseudo_csv_path, output_csv_path):
    """
    Projiziert das Pseudo-Korpus in den Einbettungsraum und evaluiert die Autorschaft.
    """
    df_pseudo = pd.read_csv(pseudo_csv_path)
    titles = df_pseudo['Titre'].values

    # Strukturelles Alignment: Fehlende Features mit 0 auffüllen
    for col in feature_cols:
        if col not in df_pseudo.columns:
            df_pseudo[col] = 0

    # Reihenfolge der Spalten exakt an den Trainings-Feature-Raum angleichen
    X_pseudo = df_pseudo[feature_cols]

    # Transformieren (Z-Standardisierung ohne Re-Fitting)
    X_scaled = scaler.transform(X_pseudo)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        embeddings = model(X_tensor).numpy()

    results = []
    for i, emb in enumerate(embeddings):
        # Euklidische Distanz zum mathematischen Asterius-Profil messen
        dist = distance.euclidean(centroid, emb)

        # Entscheidung anhand des Schwellenwerts
        is_asterius = dist <= threshold

        # Konfidenzmetrik (Je näher am Zentroid, desto valider die Zuschreibung)
        confidence = max(0, (1 - (dist / threshold))) * 100 if is_asterius else 0.0

        results.append({
            'Titre': titles[i],
            'Distanz_zu_Asterius': round(dist, 4),
            'Schwellenwert_Limit': round(threshold, 4),
            'Klassifikation': 'Asterius' if is_asterius else 'Spuria (Fremdautorschaft)',
            'Konfidenz_%': round(confidence, 2)
        })

    results_df = pd.DataFrame(results)

    # Aufsteigende Sortierung (Die stärksten Kandidaten stehen ganz oben)
    results_df = results_df.sort_values(by='Distanz_zu_Asterius')
    results_df.to_csv(output_csv_path, index=False)

    hit_count = len(results_df[results_df['Klassifikation'] == 'Asterius'])
    print(f"-> Evaluation abgeschlossen: {hit_count} von {len(df_pseudo)} Textfragmenten zugeordnet.")
    print(f"-> Ergebnisbericht gespeichert unter: '{output_csv_path}'")


# =====================================================================
# 3. AUTOMATISIERTER EINSTIEGSPUNKT (EXECUTION PIPELINE)
# =====================================================================
if __name__ == "__main__":
    print("=== STARTE INFERENZ-PHASE ===")

    # Dateipfade definieren (Analog zur README-Vorgabe)
    TRAIN_CSV = "train_features.csv"
    INFER_CSV = "inference_features.csv"
    SCALER_PATH = "scaler.pkl"
    MODEL_PATH = "siamese_asterius.pth"
    OUTPUT_CSV = "asterius_candidates.csv"

    # Validierung der Existenz kritischer Artefakte
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Kritische Datei '{SCALER_PATH}' fehlt. Trainiere das Modell zuerst via run_verification.py!")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Kritische Datei '{MODEL_PATH}' fehlt. Trainiere das Modell zuerst via run_verification.py!")
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Linguistische Matrix '{TRAIN_CSV}' fehlt. Führe erst extract_train_features.py aus!")
    if not os.path.exists(INFER_CSV):
        raise FileNotFoundError(f"Linguistische Matrix '{INFER_CSV}' fehlt. Führe erst extract_infer_features.py aus!")

    # 1. Laden des im Training angepassten Scalers
    fitted_scaler = joblib.load(SCALER_PATH)
    print(f"[OK] Scaler aus '{SCALER_PATH}' geladen.")

    # 2. Ermittlung der exakten Eingangsdimension
    df_temp = pd.read_csv(TRAIN_CSV)
    feature_count = len(df_temp.columns.drop(['Auteur', 'Titre']))
    print(f"[OK] Eingangsdimension für das neuronale Netzwerk bestimmt: {feature_count} Features.")

    # 3. Modell instanziieren und Gewichte laden
    model_instance = SiameseTabularNet(input_size=feature_count)
    model_instance.load_state_dict(torch.load(MODEL_PATH))
    model_instance.eval()
    print(f"[OK] Gewichte in SiameseTabularNet geladen. Modell befindet sich im Eval-Modus.")

    # 4. Referenzprofil extrahieren
    print("\n--- Schritt 1: Generiere Asterius-Referenzprofil (Zentroid) ---")
    asterius_centroid, dynamic_threshold, aligned_features = compute_asterius_profile(
        model=model_instance,
        scaler=fitted_scaler,
        train_csv_path=TRAIN_CSV
    )

    # 5. Unbekanntes Pseudo-Korpus klassifizieren
    print("\n--- Schritt 2: Evaluiere anonymes Pseudo-Chrysostomos-Korpus ---")
    evaluate_pseudo_corpus(
        model=model_instance,
        scaler=fitted_scaler,
        centroid=asterius_centroid,
        threshold=dynamic_threshold,
        feature_cols=aligned_features,
        pseudo_csv_path=INFER_CSV,
        output_csv_path=OUTPUT_CSV
    )

    print("\n=== PIPELINE ERFOLGREICH DURCHLAUFEN ===")