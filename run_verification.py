import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle


class PatristicTripletDataset(Dataset):
    """Lädt Anchor, Positive und Negative Samples für das Triplet-Training."""

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
                f"Kritischer Fehler: Nur {len(self.asterius_df)} Asterius-Samples gefunden. Für Triplet-Training werden mindestens 2 benötigt.")
        if len(self.hard_negatives_df) == 0 and len(self.easy_negatives_df) == 0:
            raise ValueError("Kritischer Fehler: Keine Negativ-Samples gefunden.")

        # Präkalkulation für deterministische Validierung
        if not self.is_train:
            self.val_triplets = []
            np.random.seed(
                42)
            for idx in range(len(self.asterius_df)):
                anchor = self.asterius_df[idx]
                pos_idx = np.random.choice([i for i in range(len(self.asterius_df)) if i != idx])
                positive = self.asterius_df[pos_idx]

                has_hard = len(self.hard_negatives_df) > 0
                has_easy = len(self.easy_negatives_df) > 0

                if np.random.rand() < 0.7 and has_hard:
                    neg_idx = np.random.choice(len(self.hard_negatives_df))
                    negative = self.hard_negatives_df[neg_idx]
                elif has_easy:
                    neg_idx = np.random.choice(len(self.easy_negatives_df))
                    negative = self.easy_negatives_df[neg_idx]
                else:
                    neg_idx = np.random.choice(len(self.hard_negatives_df))
                    negative = self.hard_negatives_df[neg_idx]

                self.val_triplets.append((
                    torch.FloatTensor(anchor),
                    torch.FloatTensor(positive),
                    torch.FloatTensor(negative)
                ))

    def __len__(self):
        return len(self.asterius_df)

    def __getitem__(self, idx):
        # Statische Validierung abrufen
        if not self.is_train:
            return self.val_triplets[idx]

        # Stochastisches Training
        anchor = self.asterius_df[idx]
        pos_idx = np.random.choice([i for i in range(len(self.asterius_df)) if i != idx])
        positive = self.asterius_df[pos_idx]

        has_hard = len(self.hard_negatives_df) > 0
        has_easy = len(self.easy_negatives_df) > 0

        if np.random.rand() < 0.7 and has_hard:
            neg_idx = np.random.choice(len(self.hard_negatives_df))
            negative = self.hard_negatives_df[neg_idx]
        elif has_easy:
            neg_idx = np.random.choice(len(self.easy_negatives_df))
            negative = self.easy_negatives_df[neg_idx]
        else:
            neg_idx = np.random.choice(len(self.hard_negatives_df))
            negative = self.hard_negatives_df[neg_idx]

        return torch.FloatTensor(anchor), torch.FloatTensor(positive), torch.FloatTensor(negative)


class SiameseTabularNet(nn.Module):
    def __init__(self, input_size):
        super(SiameseTabularNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.fc(x)


def train_siamese_network(csv_path):
    df = pd.read_csv(csv_path)
    feature_cols = df.columns.drop(['Auteur', 'Titre'])

    df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['Auteur'], random_state=42)

    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])

    train_dataset = PatristicTripletDataset(df_train, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = PatristicTripletDataset(df_val, is_train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    input_size = len(feature_cols)
    model = SiameseTabularNet(input_size)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for anchor, positive, negative in train_dataloader:
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
                out_a_v, out_p_v, out_n_v = model(anchor_v), model(positive_v), model(negative_v)
                val_loss = criterion(out_a_v, out_p_v, out_n_v)
                total_val_loss += val_loss.item()

        if (epoch + 1) % 10 == 0:
            avg_train_loss = total_train_loss / len(train_dataloader)
            avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model, scaler


if __name__ == "__main__":
    csv_pfad = "train_features.csv"
    print(f"Starte Training des Siamese Networks mit {csv_pfad}...")
    trainiertes_modell, angepasster_scaler = train_siamese_network(csv_pfad)

    torch.save(trainiertes_modell.state_dict(), "siamese_asterius.pth")
    print("Modellgewichte erfolgreich gespeichert: siamese_asterius.pth")

    with open("scaler.pkl", "wb") as f:
        pickle.dump(angepasster_scaler, f)
    print("Z-Standardisierungs-Scaler erfolgreich gespeichert: scaler.pkl")