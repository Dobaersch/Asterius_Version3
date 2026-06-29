import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class PatristicTripletDataset(Dataset):
    """Lädt Anchor, Positive und Negative Samples für das Triplet-Training."""
    def __init__(self, dataframe):
        self.df = dataframe
        # Trennung der Klassen
        self.asterius_df = self.df[self.df['Auteur'] == 'Asterius'].drop(columns=['Auteur', 'Titre']).values
        self.hard_negatives_df = self.df[self.df['Auteur'].isin(['Chrysostomos', 'Severian'])].drop(columns=['Auteur', 'Titre']).values
        self.easy_negatives_df = self.df[~self.df['Auteur'].isin(['Asterius', 'Chrysostomos', 'Severian'])].drop(columns=['Auteur', 'Titre']).values

    def __len__(self):
        return len(self.asterius_df)

    def __getitem__(self, idx):
        anchor = self.asterius_df[idx]
        
        # Positive: Ein anderes zufälliges Sample von Asterius
        pos_idx = np.random.choice([i for i in range(len(self.asterius_df)) if i != idx])
        positive = self.asterius_df[pos_idx]
        
        # Negative: 70% Hard Negatives (Chrysostomos/Severian), 30% Easy Negatives
        if np.random.rand() < 0.7 and len(self.hard_negatives_df) > 0:
            neg_idx = np.random.choice(len(self.hard_negatives_df))
            negative = self.hard_negatives_df[neg_idx]
        else:
            neg_idx = np.random.choice(len(self.easy_negatives_df))
            negative = self.easy_negatives_df[neg_idx]
            
        return torch.FloatTensor(anchor), torch.FloatTensor(positive), torch.FloatTensor(negative)

class SiameseTabularNet(nn.Module):
    def __init__(self, input_size):
        super(SiameseTabularNet, self).__init__()
        # MLPs eignen sich besser für tabellarische Frequenzen als CNNs/RNNs
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256), # Wichtig gegen Overfitting bei kleinen Korpura
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64) # Einbettungsraum-Dimension
        )

    def forward(self, x):
        return self.fc(x)

def train_siamese_network(csv_path):
    # 1. Daten laden und skalieren
    df = pd.read_csv(csv_path)
    feature_cols = df.columns.drop(['Auteur', 'Titre'])
    
    # Z-Standardisierung ist bei tabellarischen Frequenzen zwingend
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # 2. Dataset und DataLoader initialisieren
    dataset = PatristicTripletDataset(df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # 3. Modell und Triplet Margin Loss
    input_size = len(feature_cols)
    model = SiameseTabularNet(input_size)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 4. Trainingsschleife
    model.train()
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative in dataloader:
            optimizer.zero_grad()
            
            # Forward Pass
            out_a = model(anchor)
            out_p = model(positive)
            out_n = model(negative)
            
            # Loss berechnen
            loss = criterion(out_a, out_p, out_n)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
            
    return model, scaler