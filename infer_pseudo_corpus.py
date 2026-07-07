import pandas as pd
import numpy as np
import torch
import pickle
import scipy.spatial.distance as dist
import shap
import matplotlib.pyplot as plt
import os
import warnings
from run_verification import SiameseTabularNet

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def compute_asterius_profile(model, scaler, pca, train_csv_path):
    df_train = pd.read_csv(train_csv_path)
    feature_cols = df_train.columns.drop(['Auteur', 'Titre'])
    df_asterius = df_train[df_train['Auteur'].str.lower() == 'asterius'].copy()

    X_ast_raw = df_asterius[feature_cols].astype(float)
    X_ast_scaled = scaler.transform(X_ast_raw)
    X_ast_pca = pca.transform(X_ast_scaled)

    with torch.no_grad():
        ast_embeddings = model(torch.FloatTensor(X_ast_pca)).numpy()

    asterius_centroid = np.mean(ast_embeddings, axis=0)
    distances_to_centroid = [dist.euclidean(emb, asterius_centroid) for emb in ast_embeddings]
    dynamic_threshold = np.mean(distances_to_centroid) + 0.5 * np.std(distances_to_centroid)

    return asterius_centroid, dynamic_threshold, X_ast_raw.values, feature_cols


def run_inference(train_csv, pseudo_csv, model_path, scaler_path, pca_path, output_csv):
    print("--- Schritt 1: Lade Modelle, Scaler und PCA ---")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(pca_path, 'rb') as f:
        pca = pickle.load(f)

    pseudo_features_df = pd.read_csv(pseudo_csv)
    df_train_reference = pd.read_csv(train_csv)
    feature_cols = df_train_reference.columns.drop(['Auteur', 'Titre'])

    # Das Modell erwartet nur noch die komprimierte Dimension (z.B. 15 oder weniger)
    input_dim = pca.n_components_
    model = SiameseTabularNet(input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    print("\n--- Schritt 2: Berechne Asterius-Referenzprofil (Centroid) ---")
    asterius_centroid, dynamic_threshold, X_ast_raw, feature_names = compute_asterius_profile(model, scaler, pca,
                                                                                              train_csv)
    print(f"[Info] Dynamischer Schwellenwert berechnet: {dynamic_threshold:.4f}")

    print("\n--- Schritt 3: Evaluiere anonymes Pseudo-Chrysostomos-Korpus ---")
    titles = pseudo_features_df['Titre']
    pseudo_labels = pseudo_features_df['Auteur']

    X_pseudo_raw = pseudo_features_df[feature_cols].astype(float).values
    X_pseudo_scaled = scaler.transform(X_pseudo_raw)
    X_pseudo_pca = pca.transform(X_pseudo_scaled)

    with torch.no_grad():
        pseudo_embeddings = model(torch.FloatTensor(X_pseudo_pca)).numpy()

    # Wrapper für SHAP: Akzeptiert Original-Features, wendet Transformationen an und berechnet Distanz
    def model_distance_wrapper(X_orig):
        X_s = scaler.transform(X_orig)
        X_p = pca.transform(X_s)
        tensor_X = torch.FloatTensor(X_p)
        with torch.no_grad():
            embs = model(tensor_X).numpy()
        return np.array([dist.euclidean(emb, asterius_centroid) for emb in embs])

    print("[Info] Initialisiere SHAP KernelExplainer (auf Original-Features)...")
    background_summary = shap.kmeans(X_ast_raw, 10)
    explainer = shap.KernelExplainer(model_distance_wrapper, background_summary)

    shap_dir = "shap_explanations"
    os.makedirs(shap_dir, exist_ok=True)

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
            'Dynamic_Threshold': dynamic_threshold,
            'Classification': attribution
        })

        if distance <= (dynamic_threshold * 1.25):
            current_instance_raw = X_pseudo_raw[i:i + 1]
            shap_values = explainer.shap_values(current_instance_raw, silent=True)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, current_instance_raw, feature_names=feature_names, plot_type="bar",
                              show=False)
            plt.title(
                f"Stilometrische Metadaten: {titles.iloc[i]}\n(Positive Balken = vergrößern Distanz | Negative = verkleinern Distanz)")

            safe_title = "".join(x for x in titles.iloc[i] if x.isalnum() or x in "._- ")
            plt.savefig(os.path.join(shap_dir, f"shap_{safe_title}.png"), bbox_inches='tight', dpi=300)
            plt.close()

    df_results = pd.DataFrame(results).sort_values(by='Distance_to_Centroid')
    df_results.to_csv(output_csv, index=False)

    print(f"\n[OK] Inferenz abgeschlossen. Ergebnisse gespeichert unter: {output_csv}")
    print(f"[OK] Explainable AI Plots abgelegt in '{shap_dir}/'")


if __name__ == "__main__":
    TRAIN_CSV = "train_features.csv"
    PSEUDO_CSV = "inference_features.csv"
    MODEL_PATH = "siamese_asterius.pth"
    SCALER_PATH = "scaler.pkl"
    PCA_PATH = "pca_model.pkl"
    OUTPUT_CSV = "asterius_inference_results.csv"

    run_inference(TRAIN_CSV, PSEUDO_CSV, MODEL_PATH, SCALER_PATH, PCA_PATH, OUTPUT_CSV)