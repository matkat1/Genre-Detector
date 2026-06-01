import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
import scipy
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y.float())

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples

@torch.no_grad()

def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(y.cpu().numpy())

        total_loss += loss.item() * x.size(0)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    micro_f1 = f1_score(all_targets, all_preds, average="micro")

    return total_loss / len(loader.dataset), micro_f1

class MLP_basic(nn.Module):
    def __init__(self, input_dim=300, num_classes=30):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
          )

    def forward(self, x):
        return self.net(x)

class BiLSTMClassifier(nn.Module):
    def __init__(self, pretrained_w2v, embedding_dim, hidden_dim, num_classes=30):
        super(BiLSTMClassifier, self).__init__()
        
        weights = torch.FloatTensor(pretrained_w2v.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        out_forward = lstm_out[:, -1, :self.lstm.hidden_size]
        out_backward = lstm_out[:, 0, self.lstm.hidden_size:]
        x = torch.cat((out_forward, out_backward), dim=1)
        
        logits = self.fc(x)
        return logits

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Blok warstw, który nie zmienia wymiarowości wektora (dim -> dim)
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim), # Dobra praktyka w ResNetach
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Główny koncept ResNet: H(x) = F(x) + x
        return self.relu(self.block(x) + x)

class BiLSTM(nn.Module): # Ujednolicona nazwa
    def __init__(self, pretrained_w2v, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        weights = torch.FloatTensor(pretrained_w2v.vectors)
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, text):
        # text musi być typu torch.long (indeksy słów)
        embedded = self.embedding(text.long()) 
        lstm_out, _ = self.lstm(embedded)
        out_forward = lstm_out[:, -1, :self.lstm.hidden_size]
        out_backward = lstm_out[:, 0, self.lstm.hidden_size:]
        x = torch.cat((out_forward, out_backward), dim=1)
        return self.fc(x)

class ResNet_MLP(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, num_classes=30, num_blocks=3):
        super().__init__()
        
        # 1. Warstwa wejściowa: rzutujemy wektor (np. 300) na stały wymiar ukryty (np. 512)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Wieża bloków residualnych (tutaj dane mają cały czas rozmiar hidden_dim)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        
        # 3. Warstwa wyjściowa (klasyfikator)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.res_blocks(out)
        out = self.classifier(out)
        return out
