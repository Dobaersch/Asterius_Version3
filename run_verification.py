import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import f1_score
import joblib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Reproducibility ---
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- Device Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def optimize_verification_threshold(val_distances, val_labels):
    """
    Performs a grid search over observed validation distances to discover
    the empirical threshold that maximizes the F1-Score.
    """
    if len(val_distances) == 0:
        return 2.0, 0.0

    best_f1 = -1.0
    optimal_threshold = 0.0

    min_dist = np.min(val_distances)
    max_dist = np.max(val_distances)
    candidate_thresholds = np.linspace(min_dist, max_dist, num=200)

    for threshold in candidate_thresholds:
        # Distance less than threshold -> Asterius (Class 1)
        predictions = (val_distances < threshold).astype(int)
        current_f1 = f1_score(val_labels, predictions, pos_label=1, zero_division=0)

        if current_f1 > best_f1:
            best_f1 = current_f1
            optimal_threshold = threshold

    return float(optimal_threshold), float(best_f1)


class PatristicTripletDataset(Dataset):
    def __init__(self, X, y):
        """
        X: Numpy array of features (must be scaled and PCA-transformed).
        y: Numpy array or Pandas Series of author labels.
        """
        self.X = np.array(X)
        self.y = np.array(y)

        # Pre-compute Distance Matrix for Offline Semi-Hard Mining
        print("  [Info] Pre-computing Distance Matrix for Semi-Hard Negative Mining...")
        self.dist_matrix = euclidean_distances(self.X, self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor_features = self.X[idx]
        anchor_label = self.y[idx]

        # 1. POSITIVE SELECTION
        positive_candidates = np.where(self.y == anchor_label)[0]
        positive_candidates = positive_candidates[positive_candidates != idx]

        if len(positive_candidates) > 0:
            pos_idx = np.random.choice(positive_candidates)
        else:
            pos_idx = idx

        positive_features = self.X[pos_idx]

        # 2. SEMI-HARD NEGATIVE MINING
        negative_candidates = np.where(self.y != anchor_label)[0]
        neg_distances = self.dist_matrix[idx, negative_candidates]
        sorted_indices = np.argsort(neg_distances)

        # Select randomly from top 10 hardest negatives
        k_hardest = min(10, len(sorted_indices))
        top_k_hardest = sorted_indices[:k_hardest]
        chosen_hard_idx = np.random.choice(top_k_hardest)
        neg_idx = negative_candidates[chosen_hard_idx]

        negative_features = self.X[neg_idx]

        return torch.tensor(anchor_features, dtype=torch.float32), \
            torch.tensor(positive_features, dtype=torch.float32), \
            torch.tensor(negative_features, dtype=torch.float32)


class SiameseTabularNet(nn.Module):
    def __init__(self, input_size):
        super(SiameseTabularNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.net(x)


def train_model_instance(dataloader, model, epochs, learning_rate, verbose=False):
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    best_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    [Early Stop] Triggered at epoch {epoch + 1}")
                break
    return model


def evaluate_model_robustness(df, feature_cols):
    print("\n--- Step 1: Stratified K-Fold Cross-Validation ---")
    X = df[feature_cols].values
    y_raw = df['Auteur'].values
    # Robust normalization of author label for class comparison
    y_binary = np.array([1 if str(label).lower() == "asterius" else 0 for label in y_raw])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binary)):
        print(f"\n  [Fold {fold + 1}/5]")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y_raw[train_idx]
        y_bin_val = y_binary[val_idx]

        preprocessor = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=RANDOM_SEED))
        ])

        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)

        dataset = PatristicTripletDataset(X_train_scaled, y_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = SiameseTabularNet(input_size=X_train_scaled.shape[1]).to(device)
        train_model_instance(dataloader, model, epochs=100, learning_rate=0.0005, verbose=False)

        model.eval()
        with torch.no_grad():
            emb_train = model(torch.tensor(X_train_scaled, dtype=torch.float32).to(device)).cpu().numpy()
            emb_val = model(torch.tensor(X_val_scaled, dtype=torch.float32).to(device)).cpu().numpy()

            # Identify centroid of the target author
            ast_idx = np.where(np.array([str(y).lower() for y in y_train]) == "asterius")[0]
            if len(ast_idx) > 0:
                centroid = np.mean(emb_train[ast_idx], axis=0)
            else:
                centroid = np.zeros(emb_train.shape[1])

            val_distances = np.linalg.norm(emb_val - centroid, axis=1)
            fold_threshold, fold_f1 = optimize_verification_threshold(val_distances, y_bin_val)
            print(f"    Empirical Threshold: {fold_threshold:.4f} | Validation F1: {fold_f1:.4f}")
            fold_f1_scores.append(fold_f1)

    print(f"\n[Result] Average K-Fold Validation F1-Score: {np.mean(fold_f1_scores):.4f}")


def train_final_model(df, feature_cols):
    print("\n--- Step 2: Final Model Training & Threshold Tuning ---")
    X = df[feature_cols].values
    y_raw = df['Auteur'].values
    y_binary = np.array([1 if str(label).lower() == "asterius" else 0 for label in y_raw])

    preprocessor = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=RANDOM_SEED))
    ])

    X_scaled = preprocessor.fit_transform(X)
    print(f"  [Info] PCA reduced feature space from {X.shape[1]} to {X_scaled.shape[1]} dimensions (95% variance).")

    dataset = PatristicTripletDataset(X_scaled, y_raw)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SiameseTabularNet(input_size=X_scaled.shape[1]).to(device)
    train_model_instance(dataloader, model, epochs=150, learning_rate=0.0005, verbose=True)

    model.eval()
    with torch.no_grad():
        emb_all = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        ast_idx = np.where(np.array([str(y).lower() for y in y_raw]) == "asterius")[0]
        if len(ast_idx) > 0:
            centroid = np.mean(emb_all[ast_idx], axis=0)
        else:
            centroid = np.zeros(emb_all.shape[1])

        all_distances = np.linalg.norm(emb_all - centroid, axis=1)
        final_threshold, peak_f1 = optimize_verification_threshold(all_distances, y_binary)
        print(f"  [Export] Final Production Threshold determined as: {final_threshold:.4f}")

    # Export configuration and artifacts
    torch.save(model.state_dict(), "siamese_asterius.pth")
    joblib.dump(preprocessor, "asterius_preprocessor.pkl")

    metadata = {
        "optimal_distance_threshold": final_threshold,
        "validation_peak_f1": peak_f1,
        "pca_components_used": int(X_scaled.shape[1])
    }
    with open("model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("[Success] Model, Preprocessor, and Metadata exported successfully.")
    return model, preprocessor


if __name__ == "__main__":
    csv_path = "train_features.csv"

    print("--- Starting Siamese Network Pipeline ---")

    if not os.path.exists(csv_path):
        print(f"[Error] Dataset '{csv_path}' not found. Please run feature extraction first.")
        exit(1)

    df = pd.read_csv(csv_path)

    exclude_cols = ['Auteur', 'Titre', 'Text']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    evaluate_model_robustness(df, feature_cols)
    train_final_model(df, feature_cols)