import os
import warnings
import pandas as pd
import numpy as np
import torch
import joblib
import scipy.spatial.distance as dist
import shap
import matplotlib.pyplot as plt

# Import the architecture dynamically from your training script
from run_verification import SiameseTabularNet

# Suppress visual clutter in the terminal
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_asterius_profile(model, scaler, pca, train_csv_path):
    """
    Reconstructs the stylistic centroid (average vector) of Asterius
    and calculates the dynamic inclusion threshold based on standard deviation.
    """
    df_train = pd.read_csv(train_csv_path)

    # Isolate feature columns dynamically
    exclude_cols = ['Auteur', 'Titre', 'Text']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    # Filter for known Asterius texts
    df_asterius = df_train[df_train['Auteur'].str.contains('Asterius', case=False, na=False)].copy()

    # Apply identical transformations: Scale -> PCA -> Network
    X_ast_raw = df_asterius[feature_cols].astype(float)
    X_ast_scaled = scaler.transform(X_ast_raw)
    X_ast_pca = pca.transform(X_ast_scaled)

    with torch.no_grad():
        ast_embeddings = model(torch.FloatTensor(X_ast_pca).to(device)).cpu().numpy()

    # Calculate the centroid (geometric center) of Asterius's style
    asterius_centroid = np.mean(ast_embeddings, axis=0)

    # Calculate distances of all known Asterius chunks to their own centroid
    distances_to_centroid = [dist.euclidean(emb, asterius_centroid) for emb in ast_embeddings]

    # Dynamic Threshold: Mean distance + 0.5 * Standard Deviation
    dynamic_threshold = np.mean(distances_to_centroid) + 0.5 * np.std(distances_to_centroid)

    return asterius_centroid, dynamic_threshold, X_ast_pca, feature_cols


def run_inference():
    # --- File Paths ---
    TRAIN_CSV = "train_features.csv"
    INFERENCE_CSV = "inference_features.csv"
    MODEL_PATH = "siamese_asterius.pth"
    SCALER_PATH = "asterius_scaler.pkl"
    PCA_PATH = "asterius_pca.pkl"
    OUTPUT_CSV = "asterius_inference_results.csv"
    SHAP_DIR = "shap_plots"

    os.makedirs(SHAP_DIR, exist_ok=True)

    print("--- Step 1: Loading Trained Artifacts ---")
    try:
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
        print(f"[Success] Loaded Scaler and PCA (Dynamic Components: {pca.n_components_})")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[Error] Artifact missing: {e}. Please run run_verification.py first.")

    # Dynamically set model input size based on PCA output
    model = SiameseTabularNet(input_size=pca.n_components_).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[Success] Siamese Network weights loaded.")

    print("\n--- Step 2: Calculating Asterius Reference Profile ---")
    centroid, threshold, X_ast_pca, feature_cols = compute_asterius_profile(model, scaler, pca, TRAIN_CSV)
    print(f"Asterius Threshold set at Euclidean Distance: {threshold:.4f}")

    print("\n--- Step 3: Evaluating Anonymous Pseudo-Chrysostom Corpus ---")
    df_infer = pd.read_csv(INFERENCE_CSV)
    titles = df_infer['Titre']

    # Extract matching columns
    X_pseudo_raw = df_infer[feature_cols].astype(float)
    X_pseudo_scaled = scaler.transform(X_pseudo_raw)
    X_pseudo_pca = pca.transform(X_pseudo_scaled)

    with torch.no_grad():
        pseudo_embeddings = model(torch.FloatTensor(X_pseudo_pca).to(device)).cpu().numpy()

    # Initialize SHAP DeepExplainer using the PCA feature space
    # (Since the model input is strictly PCA components)
    tensor_ast_pca = torch.FloatTensor(X_ast_pca).to(device)
    explainer = shap.DeepExplainer(model, tensor_ast_pca)

    pca_feature_names = [f"PC_{i + 1}" for i in range(pca.n_components_)]
    results = []

    for i, emb in enumerate(pseudo_embeddings):
        distance = dist.euclidean(emb, centroid)
        attribution = "Asterius" if distance <= threshold else "Pseudo/Other"

        results.append({
            'Pseudo_Text_Titre': titles.iloc[i],
            'Distance_to_Centroid': distance,
            'Threshold': threshold,
            'Classification': attribution
        })

        # Generate SHAP explainability plot ONLY for highly likely Asterius matches
        # Allows for a 25% tolerance margin around the threshold to catch edge cases
        if distance <= (threshold * 1.25):
            current_instance_tensor = torch.FloatTensor(X_pseudo_pca[i:i + 1]).to(device)
            shap_values = explainer.shap_values(current_instance_tensor)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_pseudo_pca[i:i + 1],
                feature_names=pca_feature_names,
                plot_type="bar",
                show=False
            )
            plt.title(f"Stylometric PCA Metasynthesis: {titles.iloc[i]}\n"
                      f"(Positive Bars = Increase Distance | Negative = Decrease Distance)")

            safe_title = "".join(x for x in str(titles.iloc[i]) if x.isalnum() or x in "._- ")
            plt.savefig(os.path.join(SHAP_DIR, f"shap_{safe_title}.png"), bbox_inches='tight', dpi=300)
            plt.close()

    # Export final results
    df_results = pd.DataFrame(results).sort_values(by='Distance_to_Centroid')
    df_results.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[OK] Inference completed. Matrix saved as: '{OUTPUT_CSV}'")
    print(f"[OK] Explainable AI (SHAP) plots generated in directory: '{SHAP_DIR}/'")


if __name__ == "__main__":
    run_inference()