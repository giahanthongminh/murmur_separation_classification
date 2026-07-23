# classify_transfer_learning.py
#
# Transfer learning pipeline for murmur timing classification.
# Input : (3, 224, 224) mel spectrogram of per-segment separated murmur
# Model : ResNet18 pretrained on ImageNet → fine-tune last FC layer only
#         (optionally unfreeze all layers after warm-up)
# Split : patient-level GroupShuffleSplit (10 folds, 80/20)
#         — all segments of one patient stay in one split
#
# Why this beats MFCC+SVM after separation:
#   CNN looks at the 2D spectrogram directly and learns WHERE in time
#   the energy is — exactly the information that encodes timing class.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

DATA_ROOT    = Path.home() / "physionet.org/files/circor-heart-sound/1.0.1"
spec_dir     = DATA_ROOT / "spectrograms_dl"          # separated murmur
spec_dir_ori = DATA_ROOT / "spectrograms_original"    # original WAV
csv_path     = DATA_ROOT / "labels_dl.csv"
csv_path_ori = DATA_ROOT / "labels_original.csv"

BATCH     = 32
EPOCHS_FROZEN   = 5    # train only FC while backbone frozen
EPOCHS_FINETUNE = 15   # unfreeze all and fine-tune at lower LR
LR_FROZEN   = 1e-3
LR_FINETUNE = 1e-4
N_SPLITS    = 10
N_CLASSES   = 3

device = (torch.device("cuda") if torch.cuda.is_available()
          else torch.device("mps")  if torch.backends.mps.is_available()
          else torch.device("cpu"))
print(f"Device: {device}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SegmentDataset(Dataset):
    def __init__(self, keys, labels, data_dir):
        self.keys     = keys
        self.labels   = labels
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        arr = np.load(self.data_dir / f"{self.keys[idx]}.npy")  # (3,224,224)
        x = torch.from_numpy(arr)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Model: ResNet18 with custom head
# ---------------------------------------------------------------------------
def build_model(n_classes, freeze_backbone=True):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # Replace final FC: 512 → n_classes
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model.to(device)


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def class_weights(y_train):
    counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
    w = 1.0 / (counts + 1e-6)
    w = w / w.sum() * N_CLASSES
    return torch.tensor(w, dtype=torch.float32).to(device)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        out = model(x.to(device)).argmax(1).cpu().numpy()
        preds.extend(out)
        targets.extend(y.numpy())
    return np.array(targets), np.array(preds)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(df, data_dir):
    le     = LabelEncoder()
    keys   = df["key"].values
    y      = le.fit_transform(df["label"].values)
    groups = df["patient_id"].values

    gss = GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=42)
    accs, f1s = [], []
    all_yt, all_yp = [], []

    for fold, (tr, te) in enumerate(gss.split(keys, y, groups)):
        y_train = y[tr]
        cw = class_weights(y_train)

        train_ds = SegmentDataset(keys[tr], y_train, data_dir)
        test_ds  = SegmentDataset(keys[te], y[te],   data_dir)
        train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=False)
        test_dl  = DataLoader(test_ds,  batch_size=BATCH,
                              num_workers=0, pin_memory=False)

        # Phase 1: frozen backbone, only train FC
        model    = build_model(N_CLASSES, freeze_backbone=True)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optimizer = optim.Adam(model.fc.parameters(), lr=LR_FROZEN)

        for _ in range(EPOCHS_FROZEN):
            train_epoch(model, train_dl, optimizer, criterion)

        # Phase 2: unfreeze all, fine-tune at lower LR
        unfreeze_all(model)
        optimizer = optim.Adam(model.parameters(), lr=LR_FINETUNE,
                               weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS_FINETUNE)

        for _ in range(EPOCHS_FINETUNE):
            train_epoch(model, train_dl, optimizer, criterion)
            scheduler.step()

        yt, yp = predict(model, test_dl)
        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="weighted")
        accs.append(acc)
        f1s.append(f1)
        all_yt.extend(yt)
        all_yp.extend(yp)
        print(f"  Fold {fold+1:2d}: Acc={acc*100:.1f}%  F1={f1*100:.1f}%")

    print(f"\n  Mean Accuracy : {np.mean(accs)*100:.2f}%")
    print(f"  Mean Weighted F1: {np.mean(f1s)*100:.2f}%")
    print(classification_report(all_yt, all_yp, target_names=le.classes_))
    print(confusion_matrix(all_yt, all_yp))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Baseline: Original WAV (no separation) ---
    if csv_path_ori.exists():
        df_ori = pd.read_csv(csv_path_ori)
        print(f"\nOriginal WAV segments: {len(df_ori)}")
        print(df_ori["label"].value_counts())
        print("\n" + "="*55)
        print("ResNet18 — ORIGINAL WAV (no separation) [baseline]")
        print("="*55)
        evaluate(df_ori, spec_dir_ori)
    else:
        print("labels_original.csv not found — run generate_spectrograms_original.py first")

    # --- Proposed: Separated Murmur ---
    df_sep = pd.read_csv(csv_path)
    print(f"\nSeparated murmur segments: {len(df_sep)}")
    print(df_sep["label"].value_counts())
    print("\n" + "="*55)
    print("ResNet18 — SEPARATED MURMUR (proposed pipeline)")
    print("="*55)
    evaluate(df_sep, spec_dir)
