import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
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

        asterius_mask = df['Auteur'].str.contains('Asterius', case=False, na=False)
        self.asterius_df = df[asterius_mask][self.feature_cols].values
        other_df = df[~asterius_mask]

        hard_neg_pattern = 'Basilius|Severian|Chrysostomos|Gregor_nazianz|Gregor_nyssa'
        hard_mask = other_df['Auteur'].str.contains(hard_neg_pattern, case=False, na=False)

        self.hard_negatives_df = other_df[hard_mask][self.feature_cols].values
        self.easy_negatives_df = other_df[~hard_mask][self.feature_cols].values

    def __len__(self):
        return len(self.asterius_df)

    def __getitem__(self, idx):
        anchor = self.asterius_df[idx]

        positive_idx = np.random.choice([i for i in range(len(self.asterius_df)) if i != idx])
        positive = self.asterius_df[positive_idx]

        if len(self.hard_negatives_df) > 0 and (np.random.rand() < 0.5 or len(self.easy_negatives_df) == 0):
            neg_idx = np.random.randint(0, len(self.hard_negatives_df))
            negative = self.hard_negatives_df[neg_idx]
        else:
            neg_idx = np.random.randint(0, len(self.easy_negatives_df))
            negative = self.easy_negatives_df[neg_idx]

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


def train_model_instance(train_dataloader, model, epochs, learning_rate, verbose=False):
    """
    Core training loop for Siamese Network using Triplet Margin Loss.
    """
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        if verbose and (epoch + 1) % 20 == 0:
            avg_train_loss = total_train_loss / len(train_dataloader)
            print(f"    Epoch {epoch + 1:03d}/{epochs} | Train Triplet Loss: {avg_train_loss:.4f}")


def evaluate_pca_variance(df, feature_cols, variance_thresholds=[0.90, 0.95, 0.99], k_folds=5, epochs=100,
                          batch_size=16, learning_rate=0.001):
    """
    Acts as a manual GridSearch for PCA variance threshold utilizing a scikit-learn Pipeline.
    """
    print(f"\n[Info] Starting Dimensionality Reduction Evaluation...")
    best_variance = None
    best_f1 = -1.0

    for variance in variance_thresholds:
        print(f"\n==================================================")
        print(f"Evaluating PCA Variance Threshold: {variance * 100:.0f}%")
        print(f"==================================================")

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)
        fold_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Auteur'])):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            # Implementation of scikit-learn Pipeline
            preprocessor = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=variance, random_state=RANDOM_SEED))
            ])

            X_train_pca = preprocessor.fit_transform(train_df[feature_cols])
            X_val_pca = preprocessor.transform(val_df[feature_cols])

            pca_dim = preprocessor.named_steps['pca'].n_components_
            pca_cols = [f'PC_{i}' for i in range(pca_dim)]

            train_pca_df = pd.concat([train_df[['Auteur']].reset_index(drop=True),
                                      pd.DataFrame(X_train_pca, columns=pca_cols)], axis=1)
            val_pca_df = pd.concat([val_df[['Auteur']].reset_index(drop=True),
                                    pd.DataFrame(X_val_pca, columns=pca_cols)], axis=1)

            train_dataset = PatristicTripletDataset(train_pca_df, feature_cols=pca_cols)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            model = SiameseTabularNet(input_size=pca_dim).to(device)
            train_model_instance(train_dataloader, model, epochs, learning_rate, verbose=False)

            # k-NN Evaluation on Embeddings
            model.eval()
            with torch.no_grad():
                train_embeddings = model(torch.FloatTensor(X_train_pca).to(device)).cpu().numpy()
                val_embeddings = model(torch.FloatTensor(X_val_pca).to(device)).cpu().numpy()

            y_train_bin = train_df['Auteur'].str.contains('Asterius', case=False, na=False).astype(int).values
            y_val_bin = val_df['Auteur'].str.contains('Asterius', case=False, na=False).astype(int).values

            knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
            knn.fit(train_embeddings, y_train_bin)
            y_pred = knn.predict(val_embeddings)

            fold_f1 = f1_score(y_val_bin, y_pred, zero_division=0)
            fold_f1_scores.append(fold_f1)

        mean_f1 = np.mean(fold_f1_scores)
        print(f"[Result] {variance * 100}% Variance -> Mean F1-Score: {mean_f1:.4f} (Avg. {pca_dim} Dimensions)")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_variance = variance

    print(f"\n[Decision] Optimal PCA Variance Threshold determined as: {best_variance * 100:.0f}%")
    return best_variance


def train_final_model(df, feature_cols, optimal_variance, epochs=100, batch_size=16, learning_rate=0.001):
    """
    Trains final production model using the selected optimal variance and Pipeline.
    """
    print(f"\n[Info] Training Final Model on 100% of data (Variance: {optimal_variance * 100:.0f}%)...")

    preprocessor = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=optimal_variance, random_state=RANDOM_SEED))
    ])

    X_pca = preprocessor.fit_transform(df[feature_cols])
    pca_dim = preprocessor.named_steps['pca'].n_components_

    pca_cols = [f'PC_{i}' for i in range(pca_dim)]
    pca_df = pd.concat([df[['Auteur']].reset_index(drop=True),
                        pd.DataFrame(X_pca, columns=pca_cols)], axis=1)

    full_dataset = PatristicTripletDataset(pca_df, feature_cols=pca_cols)
    full_dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    model = SiameseTabularNet(input_size=pca_dim).to(device)
    train_model_instance(full_dataloader, model, epochs, learning_rate, verbose=True)

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

    # Step 1: Parameter Study / Grid Search equivalent for PCA Variance
    optimal_variance = evaluate_pca_variance(df, feature_cols, variance_thresholds=[0.90, 0.95, 0.98])

    # Step 2: Final Model Training using Pipeline
    trained_model, fitted_preprocessor = train_final_model(df, feature_cols, optimal_variance)

    # Export Artifacts (Pipeline consolidates Scaler & PCA)
    torch.save(trained_model.state_dict(), model_export_path)
    joblib.dump(fitted_preprocessor, preprocessor_export_path)

    print("\n--- Pipeline Execution Successful ---")
    print(f"[Success] Final Model saved to '{model_export_path}'")
    print(f"[Success] Combined Preprocessor (Scaler + PCA) saved to '{preprocessor_export_path}'")