import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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
    """

    def __init__(self, df, feature_cols):
        self.feature_cols = feature_cols

        # Separate Asterius from the rest
        self.asterius_df = df[df['Auteur'].str.contains('Asterius', case=False, na=False)][self.feature_cols].values
        other_df = df[~df['Auteur'].str.contains('Asterius', case=False, na=False)]

        # Split negative samples for Triplet Loss
        # Hard Negatives: Authors structurally close to Asterius (e.g., Basilius, Severian)
        self.hard_negatives_df = other_df[other_df['Auteur'].str.contains('Basilius|Severian', case=False, na=False)][
            self.feature_cols].values
        self.easy_negatives_df = other_df[~other_df['Auteur'].str.contains('Basilius|Severian', case=False, na=False)][
            self.feature_cols].values

    def __len__(self):
        return len(self.asterius_df)

    def __getitem__(self, idx):
        anchor = self.asterius_df[idx]

        # Select a random OTHER Asterius sample (Positive)
        positive_idx = np.random.choice([i for i in range(len(self.asterius_df)) if i != idx])
        positive = self.asterius_df[positive_idx]

        # Select a negative sample (Hard or Easy Negative)
        if len(self.hard_negatives_df) > 0 and (np.random.rand() < 0.5 or len(self.easy_negatives_df) == 0):
            neg_idx = np.random.randint(0, len(self.hard_negatives_df))
            negative = self.hard_negatives_df[neg_idx]
        else:
            neg_idx = np.random.randint(0, len(self.easy_negatives_df))
            negative = self.easy_negatives_df[neg_idx]

        # Gaussian Noise injection is strictly removed to preserve discrete stylistic features
        return torch.FloatTensor(anchor.copy()), torch.FloatTensor(positive.copy()), torch.FloatTensor(negative.copy())

class SiameseTabularNet(nn.Module):
    """
    Siamese neural network architecture for tabular stylometric feature data.
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


def train_siamese_network(csv_path, epochs=100, batch_size=16, learning_rate=0.001):
    """
    Loads data, applies dynamic PCA, and trains the Siamese Network with Triplet Loss.
    """
    df = pd.read_csv(csv_path)

    # Identify feature columns dynamically (excluding metadata columns)
    exclude_cols = ['Auteur', 'Titre', 'Text']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Train/Validation split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Auteur'], random_state=RANDOM_SEED)

    # 1. Feature Scaling (Z-Standardization is mandatory before PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_val_scaled = scaler.transform(val_df[feature_cols])

    # 2. Dynamic PCA: Dimensionality Reduction based on Variance
    # Retaining 95% of the stylistic variance instead of an arbitrary fixed number
    pca = PCA(n_components=0.95, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    n_components_retained = pca.n_components_
    print(f"[Info] PCA dynamically reduced features to {n_components_retained} components (95% variance retained).")

    # Override dataframes with PCA components
    train_pca_df = train_df[['Auteur', 'Titre']].copy()
    val_pca_df = val_df[['Auteur', 'Titre']].copy()

    pca_cols = [f'PC_{i}' for i in range(n_components_retained)]

    train_pca_features = pd.DataFrame(X_train_pca, columns=pca_cols, index=train_df.index)
    val_pca_features = pd.DataFrame(X_val_pca, columns=pca_cols, index=val_df.index)

    train_pca_df = pd.concat([train_pca_df, train_pca_features], axis=1)
    val_pca_df = pd.concat([val_pca_df, val_pca_features], axis=1)

    # 3. Initialize Datasets using the dynamically generated PCA columns
    train_dataset = PatristicTripletDataset(train_pca_df, feature_cols=pca_cols)
    val_dataset = PatristicTripletDataset(val_pca_df, feature_cols=pca_cols)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize the model dynamically based on PCA dimension output
    model = SiameseTabularNet(input_size=n_components_retained).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("[Info] Starting Training Process...")

    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for anchor, positive, negative in train_dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            out_a, out_p, out_n = model(anchor), model(positive), model(negative)

            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for anchor_v, positive_v, negative_v in val_dataloader:
                anchor_v, positive_v, negative_v = anchor_v.to(device), positive_v.to(device), negative_v.to(device)
                out_a_v, out_p_v, out_n_v = model(anchor_v), model(positive_v), model(negative_v)

                val_loss = criterion(out_a_v, out_p_v, out_n_v)
                total_val_loss += val_loss.item()

        if (epoch + 1) % 10 == 0:
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
            print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Return model to CPU to ensure safe persistence
    return model.cpu(), scaler, pca


if __name__ == "__main__":
    # Standardized English variables
    csv_path = "train_features.csv"
    model_export_path = "siamese_asterius.pth"
    scaler_export_path = "asterius_scaler.pkl"
    pca_export_path = "asterius_pca.pkl"

    print("--- Starting Siamese Network Pipeline ---")
    print(f"[Info] Loading training features from '{csv_path}'")

    trained_model, fitted_scaler, fitted_pca = train_siamese_network(csv_path, epochs=100)

    # Safe and complete artifact persistence using joblib and torch
    torch.save(trained_model.state_dict(), model_export_path)
    joblib.dump(fitted_scaler, scaler_export_path)
    joblib.dump(fitted_pca, pca_export_path)

    print("\n--- Pipeline Execution Successful ---")
    print(f"[Success] Model saved to '{model_export_path}'")
    print(f"[Success] Scaler saved to '{scaler_export_path}'")
    print(f"[Success] PCA mapping saved to '{pca_export_path}'")