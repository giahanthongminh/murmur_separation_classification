# classify_cnn.py
# CNN classifier for murmur timing classification
# Input: Mel Spectrogram (64x64) from separated murmur WAV files
# Patient-level 80/20 split to avoid data leakage

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from config import FEATURE_OUTPUT_DIR
from src.data_validation import validate_dataset

spectrogram_dir = FEATURE_OUTPUT_DIR / "spectrograms"

# Use MPS (Apple Silicon GPU) if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class MurmurDataset(Dataset):
    def __init__(self, keys, labels):
        self.keys = keys
        self.labels = labels

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        spec = np.load(spectrogram_dir / f"{self.keys[idx]}.npy")
        # Normalize to [0, 1]
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        # Add channel dim → (1, 64, 64)
        x = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class MurmurCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def predict(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).argmax(dim=1).cpu().numpy()
            preds.extend(out)
            targets.extend(y.numpy())
    return np.array(targets), np.array(preds)


def evaluate_cnn(df, n_splits=10, epochs=30):
    le = LabelEncoder()
    keys = df["key"].values
    y = le.fit_transform(df["label"].values)
    groups = df["patient_id"].values

    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    accs, f1s = [], []
    all_y_test, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(gss.split(keys, y, groups)):
        train_set = MurmurDataset(keys[train_idx], y[train_idx])
        test_set  = MurmurDataset(keys[test_idx],  y[test_idx])

        # Class weights to handle imbalance
        class_counts = np.bincount(y[train_idx])
        weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader  = DataLoader(test_set,  batch_size=32)

        model = MurmurCNN(n_classes=3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(weight=weights)

        for _ in range(epochs):
            train_epoch(model, train_loader, optimizer, criterion)

        y_test, y_pred = predict(model, test_loader)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="weighted"))
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"  Fold {fold+1}: Acc={accs[-1]*100:.1f}%")

    print(f"\n  Accuracy:    {np.mean(accs)*100:.2f}%")
    print(f"  Weighted F1: {np.mean(f1s)*100:.2f}%")
    print(classification_report(all_y_test, all_y_pred,
                                target_names=le.classes_))
    print(confusion_matrix(all_y_test, all_y_pred))


validate_dataset()
df = pd.read_csv(FEATURE_OUTPUT_DIR / "labels_cnn.csv")

print("\n" + "="*50)
print("CNN — Separated Murmur (Mel Spectrogram)")
print("="*50)
evaluate_cnn(df)
