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



