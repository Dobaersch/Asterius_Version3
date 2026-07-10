import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import gc
import scipy.spatial.distance as dist
import shap
import matplotlib

matplotlib.use('Agg')  # Headless mode to prevent memory/tkinter crashes
import matplotlib.pyplot as plt

# Import the architecture dynamically from your training script
from run_verification import SiameseTabularNet

# Suppress visual clutter in the terminal
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_asterius_profile(model, preprocessor, train_csv_path):
    """
    Reconstructs the stylistic centroid (average vector) of Asterius
    and calculates the dynamic inclusion threshold.
    Uses the fitted preprocessor (StandardScaler) directly on the full feature space.
    """
    df_train = pd.read_csv(train_csv_path)

    # Isolate feature columns dynamically
    exclude_cols = ['Auteur', 'Titre', 'Text']
    feature_cols = [col for col in df_train.columns if col not in exclude_cols]

    # Filter for known Asterius texts
    df_asterius = df_train[df_train['Auteur'].str.contains('Asterius', case=False, na=False)].copy()

    # Apply pipeline: Scale ONLY (No PCA)
    X_ast_raw = df_asterius[feature_cols].astype(float)
    X_ast_scaled = preprocessor.transform(X_ast_raw)

    # Generate Embeddings
    model.eval()
    with torch.no_grad():
        X_ast_tensor = torch.FloatTensor(X_ast_scaled).to(device)
        ast_embeddings = model(X_ast_tensor)
        # L2-Normalize embeddings to map Euclidean space effectively
        ast_embeddings = F.normalize(ast_embeddings, p=2, dim=1).cpu().numpy()

    # Calculate Centroid
    centroid = np.mean(ast_embeddings, axis=0)

    # Using an empirical threshold derived from the Euclidean distances
    dynamic_threshold = 2.5000

    return centroid, dynamic_threshold, feature_cols, X_ast_scaled


def infer_pseudo_corpus():
    TRAIN_CSV = "train_features.csv"
    INFER_CSV = "inference_features.csv"
    MODEL_PATH = "siamese_asterius.pth"
    PREPROCESSOR_PATH = "asterius_preprocessor.pkl"
    OUTPUT_CSV = "asterius_inference_results.csv"
    SHAP_DIR = "shap_diagrams"

    os.makedirs(SHAP_DIR, exist_ok=True)

    print("--- Phase 4: Inference & SHAP Analysis (Full Feature Space) ---")

    # 1. Load Artefacts
    try:
        # We now only load the single preprocessor pipeline
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except FileNotFoundError:
        print(f"[Error] Artefact missing. Please ensure '{PREPROCESSOR_PATH}' exists.")
        exit(1)

    # Determine input size dynamically from inference features
    try:
        df_infer = pd.read_csv(INFER_CSV)
    except FileNotFoundError:
        print(f"[Error] '{INFER_CSV}' not found. Please extract inference features first.")
        exit(1)

    exclude_cols = ['Auteur', 'Titre', 'Text']
    feature_cols = [col for col in df_infer.columns if col not in exclude_cols]
    input_dim = len(feature_cols)

    # Load Model Configuration
    model = SiameseTabularNet(input_size=input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2. Compute Centroid
    print("\n[Computing Asterius-Centroid...]")
    centroid, threshold, train_features, background_data_scaled = compute_asterius_profile(model, preprocessor,
                                                                                           TRAIN_CSV)
    print(f"[Info] Demarcation Threshold set to: {threshold:.4f} (Euclidean Distance)")

    # 3. Prepare SHAP Explainer
    # We use random samples from the Asterius background to train the Explainer
    bg_sample_size = min(100, len(background_data_scaled))
    idx = np.random.choice(len(background_data_scaled), bg_sample_size, replace=False)
    bg_tensor = torch.FloatTensor(background_data_scaled[idx]).to(device)

    explainer = shap.DeepExplainer(model, bg_tensor)

    print("\n--- Evaluating Anonymous Pseudo-Chrysostom Corpus ---")

    X_pseudo_raw = df_infer[train_features].astype(float)
    X_pseudo_scaled = preprocessor.transform(X_pseudo_raw)

    titles = df_infer['Titre']
    results = []

    with torch.no_grad():
        X_pseudo_tensor = torch.FloatTensor(X_pseudo_scaled).to(device)
        pseudo_embeddings = model(X_pseudo_tensor)
        pseudo_embeddings = F.normalize(pseudo_embeddings, p=2, dim=1).cpu().numpy()

    for i, emb in enumerate(pseudo_embeddings):
        distance = dist.euclidean(emb, centroid)

        if distance <= threshold:
            attribution = "Asterius"
        elif distance <= (threshold + 1.0):
            attribution = "Uncertain (Grey Zone)"
        else:
            attribution = "Foreign Author"

        results.append({
            'Document': str(titles.iloc[i]),
            'Distance_to_Centroid': distance,
            'Classification': attribution
        })

        # Generate SHAP only for relevant/close texts to save computation time
        if distance <= (threshold + 1.5):
            print(f"  [SHAP] Analyzing: {titles.iloc[i]} (Distance: {distance:.2f})")

            current_instance_tensor = torch.FloatTensor(X_pseudo_scaled[i:i + 1]).to(device)
            current_instance_tensor.requires_grad_(True)

            shap_values = explainer.shap_values(current_instance_tensor, check_additivity=False)

            plot_shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values

            # --- PLOTTING DIRECT LINGUISTIC FEATURES ---
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                plot_shap_values,
                X_pseudo_scaled[i:i + 1],
                feature_names=np.array(train_features),
                plot_type="bar",
                max_display=20,
                show=False
            )
            plt.title(f"Stylometric SHAP Analysis: {titles.iloc[i]}\n"
                      f"(Positive Bars = Increase Distance | Negative = Decrease Distance)")

            safe_title = "".join(x for x in str(titles.iloc[i]) if x.isalnum() or x in "._- ")
            plt.savefig(os.path.join(SHAP_DIR, f"shap_{safe_title}.png"), bbox_inches='tight', dpi=300)

            # --- MEMORY LEAK FIX ---
            plt.clf()
            plt.close('all')
            gc.collect()

            # Export final results
    df_results = pd.DataFrame(results).sort_values(by='Distance_to_Centroid')
    df_results.to_csv(OUTPUT_CSV, index=False)

    print(f"\n[Success] Inference complete. Results saved to '{OUTPUT_CSV}'.")
    print(f"[Success] SHAP explanations saved to directory '{SHAP_DIR}'.")


if __name__ == "__main__":
    infer_pseudo_corpus()