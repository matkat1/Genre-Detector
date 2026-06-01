import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import re
import scipy
import ast
import gensim.downloader
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv("/content/drive/MyDrive/data_goodreads.csv")
df = df.head(300000)

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return None

def parse_genres(x):
    if not isinstance(x, str):
        return []
    return re.findall(r'"(.*?)"', x)

def encode_genres(genres):
    vec = np.zeros(30, dtype=np.float32)
    for g in genres:
        if g in genre_to_idx:
            vec[genre_to_idx[g]] = 1.0
    return vec

df["genres"] = df["genres"].apply(parse_genres)
top30 = df["genres"].explode().value_counts().head(30).index.tolist()
genre_to_idx = {g: i for i, g in enumerate(top30)}
y = np.stack(df["genres"].apply(encode_genres))

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_loader(X, y):
    dataset = TextDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader

def load_X():
    container = np.load('/content/drive/MyDrive/Genre_Detector/data/w2v_300_emb.npz')
    X = container['w2v_idxs']
    X = X[:300000]
    return X

class BiLSTMClassifier(nn.Module):
    def __init__(self, pretrained_w2v, embedding_dim, hidden_dim, num_classes):
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

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

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
    micro_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)

    return total_loss / len(loader.dataset), micro_f1

def experiment(X, y, pretrained_w2v_name='glove-wiki-gigaword-100'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    w2v_model = gensim.downloader.load(pretrained_w2v_name)
    
    model = BiLSTMClassifier(
        pretrained_w2v=w2v_model,
        embedding_dim=w2v_model.vector_size,
        hidden_dim=128,
        num_classes=30
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pos_counts = y.sum(axis=0)
    neg_counts = len(y) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_loader = create_loader(X_train, y_train)
    test_loader = create_loader(X_test, y_test)

    print("Rozpoczęcie treningu...")
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Micro-F1: {val_f1:.4f}")

#stara
X = load_X()
experiment(X, y, pretrained_w2v_name='word2vec-google-news-300')