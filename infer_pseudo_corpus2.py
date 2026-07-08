import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SiameseTabularNet(nn.Module):
    """
    Architecture MUST perfectly match the training script.
    """
    def __init__(self, input_size):
        super(SiameseTabularNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.fc(x)

def run_inference(inference_csv_path, output_csv_path):
    """
    Loads artifacts and classifies unseen documents.
    """
    # 1. Path Definitions for Artifacts
    model_path = "siamese_asterius.pth"
    preprocessor_path = "asterius_preprocessor.pkl"
    knn_path = "asterius_knn.pkl"

    print("--- Starting Stylometric Inference Pipeline ---")
    
    # 2. Load New Dataset
    try:
        df_infer = pd.read_csv(inference_csv_path)
        print(f"[Info] Loaded {len(df_infer)} documents for inference.")
    except FileNotFoundError:
        print(f"[Error] Dataset '{inference_csv_path}' not found.")
        exit(1)

    # Maintain metadata to identify documents later
    metadata_cols = ['Auteur', 'Titre']
    existing_meta = [col for col in metadata_cols if col in df_infer.columns]
    feature_cols = [col for col in df_infer.columns if col not in ['Auteur', 'Titre', 'Text']]

    # 3. Load Exported Artifacts
    print("[Info] Loading Preprocessor and ML Artifacts...")
    preprocessor = joblib.load(preprocessor_path)
    knn_classifier = joblib.load(knn_path)

    # Reconstruct Siamese Network dynamically based on PCA dimensions
    pca_dim = preprocessor.named_steps['pca'].n_components_
    model = SiameseTabularNet(input_size=pca_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Data Transformation Pipeline
    print("[Info] Transforming raw features to stylometric embeddings...")
    
    # Apply exactly the same Scaling and PCA
    X_infer_pca = preprocessor.transform(df_infer[feature_cols])

    # Extract 32-D Embeddings using the Siamese Network
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_infer_pca).to(device)
        embeddings = model(X_tensor).cpu().numpy()

    # 5. k-NN Classification
    print("[Info] Executing k-NN Cosine Classification...")
    predictions = knn_classifier.predict(embeddings)
    probabilities = knn_classifier.predict_proba(embeddings)

    # 6. Structuring Output
    results_df = df_infer[existing_meta].copy() if existing_meta else pd.DataFrame()
    results_df['Predicted_Author'] = ["Asterius" if pred == 1 else "Other/Pseudo" for pred in predictions]
    
    # Extract probability for the positive class (Asterius)
    asterius_class_idx = list(knn_classifier.classes_).index(1)
    results_df['Confidence_Asterius'] = probabilities[:, asterius_class_idx]

    # Save Results
    results_df.to_csv(output_csv_path, index=False)
    print(f"\n[Success] Inference complete. Results saved to '{output_csv_path}'")
    
    # Display Summary
    asterius_count = sum(predictions)
    print(f"\n--- Inference Summary ---")
    print(f"Total documents analyzed : {len(df_infer)}")
    print(f"Attributed to Asterius   : {asterius_count}")
    print(f"Attributed to Other      : {len(df_infer) - asterius_count}")


if __name__ == "__main__":
    # Define input and output files
    unseen_texts_csv = "pseudo_corpus_features.csv"
    results_csv = "inference_results.csv"
    
    run_inference(unseen_texts_csv, results_csv)