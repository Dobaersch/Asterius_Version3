import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle

# --- METHODISCHE ABSICHERUNG: DETERMINISMUS ---
# Setzen der globalen Seeds für wissenschaftliche Reproduzierbarkeit.
# Garantiert, dass das Modell bei jedem Durchlauf exakt dieselben Vektorräume aufspannt.
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Hardware-Beschleunigung aktivieren, falls vorhanden
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PatristicTripletDataset(Dataset):
    """Lädt Anchor, Positive und Negative Samples für das Triplet-Training inkl. Data Augmentation."""

    def __init__(self, dataframe, is_train=True):
        self.df = dataframe
        self.is_train = is_train

        auteurs_lower = self.df['Auteur'].str.lower()
        self.asterius_df = self.df[auteurs_lower == 'asterius'].drop(columns=['Auteur', 'Titre']).values
        self.hard_negatives_df = self.df[auteurs_lower.isin(['chrysostomos', 'basilius'])].drop(
            columns=['Auteur', 'Titre']).values
        self.easy_negatives_df = self.df[~auteurs_lower.isin(['asterius', 'chrysostomos', 'basilius'])].drop(
            columns=['Auteur', 'Titre']).values

        if len(self.asterius_df) < 2:
            raise ValueError(
                f"Kritischer Fehler: Nur {len(self.asterius_df)} Asterius-Samples gefunden. Für Triplet-Training werden mindestens 2 benötigt."
            )
        if len(self.hard_negatives_df) == 0 and len(self.easy_negatives_df) == 0:
            raise ValueError("Kritischer Fehler: Keine negativen Samples gefunden.")

    def __len__(self):
        return len(self.asterius_df)

    def __getitem__(self, idx):
        anchor = self.asterius_df[idx]

        # Wähle ein zufälliges ANDERES Asterius-Sample
        positive_idx = np.random.choice([i for i in range(len(self.asterius_df)) if i != idx])
        positive = self.asterius_df[positive_idx]

        if len(self.hard_negatives_df) > 0 and (np.random.rand() < 0.5 or len(self.easy_negatives_df) == 0):
            neg_idx = np.random.randint(0, len(self.hard_negatives_df))
            negative = self.hard_negatives_df[neg_idx]
        else:
            neg_idx = np.random.randint(0, len(self.easy_negatives_df))
            negative = self.easy_negatives_df[neg_idx]

        # --- DATA AUGMENTATION ---
        # Addiere leichtes Gaußsches Rauschen während des Trainings,
        # um Overfitting/Memorisation mathematisch unmöglich zu machen.
        if self.is_train:
            noise_factor = 0.05
            anchor = anchor + np.random.normal(0, noise_factor, anchor.shape)
            positive = positive + np.random.normal(0, noise_factor, positive.shape)
            negative = negative + np.random.normal(0, noise_factor, negative.shape)

        return torch.FloatTensor(anchor), torch.FloatTensor(positive), torch.FloatTensor(negative)


class SiameseTabularNet(nn.Module):
    """Mikro-Architektur, um Auswendiglernen bei extrem kleinen Datensätzen zu verhindern."""
    def __init__(self, input_dim):
        super(SiameseTabularNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(16, 8)
        )

    def forward(self, x):
        return self.network(x)


class TripletLoss(nn.Module):
    """Berechnet den Hinge-Loss zwischen Anchor, Positive und Negative."""

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = self.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def train_siamese_network(csv_path, epochs=50, batch_size=16, learning_rate=0.001):
    df = pd.read_csv(csv_path)

    print(f"[Info] Nutze Recheneinheit: {device}")

    # --- STATISTISCHER FIX: STRATIFIED SPLIT ---
    # Erzeugt eine Hilfsspalte (True/False), ob der Text von Asterius stammt.
    # Die Stratifizierung garantiert, dass exakt das gleiche Verhältnis von Asterius-Texten
    # im Trainings- und Validierungs-Set landet. Beugt Laufzeit-Crashes vor.
    is_asterius = (df['Auteur'].str.lower() == 'asterius')

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=is_asterius
    )

    feature_cols = df.columns.drop(['Auteur', 'Titre'])

    # --- TYP-KONVERTIERUNG ---
    train_df[feature_cols] = train_df[feature_cols].astype(float)
    val_df[feature_cols] = val_df[feature_cols].astype(float)

    # 1. Skalierung (Zwingend erforderlich VOR der PCA)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[feature_cols])
    X_val_scaled = scaler.transform(val_df[feature_cols])

    # 2. PCA: Dimensionsreduktion
    # Wir zwingen die hunderten Features auf maximal 15 essentielle Hauptkomponenten herunter
    n_components = min(15, len(feature_cols))
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    # Wir überschreiben die alten, breiten Dataframes mit den neuen, schmalen PCA-Vektoren
    train_pca_df = train_df[['Auteur', 'Titre']].copy()
    val_pca_df = val_df[['Auteur', 'Titre']].copy()

    pca_cols = [f'PC_{i}' for i in range(n_components)]
    train_pca_df[pca_cols] = X_train_pca
    val_pca_df[pca_cols] = X_val_pca

    # Datasets mit komprimierten PCA-Daten aufrufen
    train_dataset = PatristicTripletDataset(train_pca_df, is_train=True)
    val_dataset = PatristicTripletDataset(val_pca_df, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialisierung
    input_dim = len(pca_cols)  # Das Netz hat nun einen extrem reduzierten Input!
    model = SiameseTabularNet(input_dim).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    print("\n--- Starte Epochen ---")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for anchor, positive, negative in train_dataloader:
            # Daten auf GPU verlagern, falls vorhanden
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            out_a, out_p, out_n = model(anchor), model(positive), model(negative)
            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for anchor_v, positive_v, negative_v in val_dataloader:
                anchor_v, positive_v, negative_v = anchor_v.to(device), positive_v.to(device), negative_v.to(device)
                out_a_v, out_p_v, out_n_v = model(anchor_v), model(positive_v), model(negative_v)
                val_loss = criterion(out_a_v, out_p_v, out_n_v)
                total_val_loss += val_loss.item()

        if (epoch + 1) % 10 == 0:
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
            print(f"Epoch {epoch + 1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Für die spätere Inferenz (die auf CPU laufen könnte) das Modell zurück auf die CPU holen
    return model.cpu(), scaler, pca


if __name__ == "__main__":
    csv_pfad = "train_features.csv"
    model_export_path = "siamese_asterius.pth"
    scaler_export_path = "scaler.pkl"

    print(f"Starte Training des Siamese Networks mit Daten aus '{csv_pfad}'...")

    trained_model, fitted_scaler, fitted_pca = train_siamese_network(csv_pfad, epochs=100)

    torch.save(trained_model.state_dict(), model_export_path)
    with open(scaler_export_path, 'wb') as f:
        pickle.dump(fitted_scaler, f)

    pca_export_path = "pca_model.pkl"
    with open(pca_export_path, 'wb') as f:
        pickle.dump(fitted_pca, f)

    print(f"\n[OK] Training abgeschlossen.")
    print(f"[OK] Modellgewichte exportiert nach: {model_export_path}")
    print(f"[OK] Scaler-Matrix exportiert nach: {scaler_export_path}")
    print(f"[OK] PCA-Matrix exportiert nach: {pca_export_path}")