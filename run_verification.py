import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
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


class PatristicTripletDataset(Dataset):
    """
    Dataset for Siamese Network generating Anchors, Positives, and Negatives.
    Learns a universal stylometric embedding space by sampling anchors from ALL authors.
    Applies Gaussian noise augmentation to prevent memorisation.
    """

    def __init__(self, df, feature_cols, noise_std=0.02, training=True):
        self.feature_cols = feature_cols
        self.noise_std = noise_std
        self.training = training

        self.df = df.reset_index(drop=True)
        self.authors = self.df['Auteur'].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Anchor
        anchor_row = self.df.iloc[idx]
        anchor_author = anchor_row['Auteur']
        anchor = anchor_row[self.feature_cols].values.astype(np.float32)

        # 2. Positive (Same author)
        positive_candidates = self.df[self.df['Auteur'] == anchor_author]
        if len(positive_candidates) > 1:
            positive_candidates = positive_candidates.drop(positive_candidates.index[positive_candidates.index == idx])

        positive = positive_candidates.sample(1).iloc[0][self.feature_cols].values.astype(np.float32)

        # 3. Negative (Different author)
        negative_candidates = self.df[self.df['Auteur'] != anchor_author]
        negative = negative_candidates.sample(1).iloc[0][self.feature_cols].values.astype(np.float32)

        # Augmentation
        if self.training:
            anchor += np.random.normal(0, self.noise_std, anchor.shape)
            positive += np.random.normal(0, self.noise_std, positive.shape)
            negative += np.random.normal(0, self.noise_std, negative.shape)

        return torch.FloatTensor(anchor), torch.FloatTensor(positive), torch.FloatTensor(negative)


class SiameseTabularNet(nn.Module):
    """
    A deep neural network to compress stylometric features into a highly dense embedding space.
    Uses LayerNorm and Dropout for regularisation.
    """

    def __init__(self, input_size, embedding_size=8):
        super(SiameseTabularNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),

            nn.Linear(32, embedding_size)
        )

    def forward(self, x):
        return self.net(x)


def train_model_instance(dataloader, model, epochs, learning_rate, verbose=False):
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 15

    for epoch in range(epochs):
        model.train()
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

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        if patience_counter >= patience_limit:
            if verbose:
                print(f"    [Early Stop] No improvement for {patience_limit} epochs at epoch {epoch + 1}.")
            break

    return model


def evaluate_model_robustness(df, feature_cols):
    """
    Evaluates the Siamese Network directly on scaled features without PCA.
    Uses Silhouette Score to measure clustering quality.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_silhouettes = []

    print(f"\n{'=' * 50}\nEvaluating Siamese Network (Full Feature Space)\n{'=' * 50}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Auteur'])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # 1. Pipeline: ONLY Scaler, NO PCA
        preprocessor = Pipeline([
            ('scaler', StandardScaler())
        ])

        X_train_raw = train_df[feature_cols].astype(float)
        X_val_raw = val_df[feature_cols].astype(float)

        preprocessor.fit(X_train_raw)

        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()

        train_df_scaled[feature_cols] = preprocessor.transform(X_train_raw)
        val_df_scaled[feature_cols] = preprocessor.transform(X_val_raw)

        current_dim = train_df_scaled[feature_cols].shape[1]

        # 2. Dataset Setup
        train_dataset = PatristicTripletDataset(train_df_scaled, feature_cols)

        # 3. Model Training
        model = SiameseTabularNet(input_size=current_dim).to(device)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Train briefly for evaluation
        train_model_instance(train_dataloader, model, epochs=25, learning_rate=0.001, verbose=False)

        # 4. Generate Embeddings for validation
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(val_df_scaled[feature_cols].values).to(device)
            val_embeddings = model(X_val_tensor)
            val_embeddings = F.normalize(val_embeddings, p=2, dim=1).cpu().numpy()

        # 5. Evaluate Distances
        y_val_binary = np.where(val_df['Auteur'].str.contains('Asterius', case=False), 1, 0)

        if len(np.unique(y_val_binary)) > 1:
            sil_score = silhouette_score(val_embeddings, y_val_binary, metric='euclidean')
            fold_silhouettes.append(sil_score)
        else:
            fold_silhouettes.append(0.0)

    # These lines are now correctly indented inside the function!
    mean_silhouette = np.mean(fold_silhouettes)
    print(f"[Result] Mean Silhouette Score across folds: {mean_silhouette:.4f} (using {current_dim} input features)")


def train_final_model(df, feature_cols):
    """
    Trains the final production model on 100% of the data WITHOUT PCA.
    """
    print(f"\n[Info] Training Final Model on 100% of data (Full Feature Space)...")

    preprocessor = Pipeline([
        ('scaler', StandardScaler())
    ])

    X_raw = df[feature_cols].astype(float)
    preprocessor.fit(X_raw)

    df_scaled = df.copy()
    df_scaled[feature_cols] = preprocessor.transform(X_raw)

    input_dim = df_scaled[feature_cols].shape[1]

    # Initialize Dataset with scaled data
    full_dataset = PatristicTripletDataset(df_scaled, feature_cols)
    batch_size = min(32, len(full_dataset))
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    model = SiameseTabularNet(input_size=input_dim).to(device)

    # Train for up to 100 epochs, Early Stopping will cut it short when optimal
    train_model_instance(full_dataloader, model, epochs=100, learning_rate=0.0005, verbose=True)

    return model.cpu(), preprocessor


if __name__ == "__main__":
    csv_path = "train_features.csv"
    model_export_path = "siamese_asterius.pth"
    preprocessor_export_path = "asterius_preprocessor.pkl"

    print("--- Starting Siamese Network Pipeline ---")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[Error] Dataset '{csv_path}' not found. Please ensure the file exists.")
        exit(1)

    exclude_cols = ['Auteur', 'Titre', 'Text']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Step 1: Model Evaluation (replaces PCA Variance Search)
    evaluate_model_robustness(df, feature_cols)

    # Step 2: Final Model Training using full feature space
    trained_model, fitted_preprocessor = train_final_model(df, feature_cols)

    # Export Artifacts
    torch.save(trained_model.state_dict(), model_export_path)
    joblib.dump(fitted_preprocessor, preprocessor_export_path)

    print("\n--- Pipeline Execution Successful ---")
    print(f"[Success] Final Model saved to '{model_export_path}'")
    print(f"[Success] Scaler saved to '{preprocessor_export_path}'")