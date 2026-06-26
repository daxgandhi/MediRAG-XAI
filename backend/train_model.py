"""
Disease Model Training Script
Uses REAL Kaggle Disease-Symptom Dataset (4,920 rows, 132 symptoms, 41 diseases)
Source: itachi9604/healthcare-chatbot GitHub / Kaggle kaushil268
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "datasets"
MODELS_DIR = BASE_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# ─── Dataset Loading ──────────────────────────────────────────────────────────
def load_dataset():
    train_path = DATA_DIR / "Training.csv"
    test_path  = DATA_DIR / "Testing.csv"

    if not train_path.exists():
        print("[ERROR] Training.csv not found! Run download commands first.")
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Drop unnamed/extra columns
    df_train = df_train.loc[:, ~df_train.columns.str.contains("^Unnamed")]
    df_test  = df_test.loc[:,  ~df_test.columns.str.contains("^Unnamed")]

    print(f"[OK] Training set: {df_train.shape[0]} rows × {df_train.shape[1]} cols")
    print(f"[OK] Testing  set: {df_test.shape[0]} rows × {df_test.shape[1]} cols")
    print(f"[OK] Diseases found: {df_train['prognosis'].nunique()}")

    return df_train, df_test

# ─── PyTorch Model ────────────────────────────────────────────────────────────
class DiseaseNet(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.network(x)

# ─── Training Pipeline ────────────────────────────────────────────────────────
def train():
    print("\n" + "="*60)
    print("  MEDIRAG-XAI — Disease Prediction Model Training")
    print("  Dataset: Kaggle Disease-Symptom (itachi9604)")
    print("="*60 + "\n")

    # 1. Load data
    df_train, df_test = load_dataset()

    # 2. Features and Labels
    feature_cols = [c for c in df_train.columns if c != "prognosis"]
    X_train_all  = df_train[feature_cols].values.astype(np.float32)
    X_test_all   = df_test[feature_cols].values.astype(np.float32)

    # 3. Encode labels
    le = LabelEncoder()
    y_train_all = le.fit_transform(df_train["prognosis"])
    y_test_all  = le.transform(df_test["prognosis"])

    num_classes = len(le.classes_)
    input_size  = X_train_all.shape[1]
    print(f"[INFO] Input features : {input_size}")
    print(f"[INFO] Disease classes: {num_classes}")
    print(f"[INFO] Classes: {list(le.classes_)}\n")

    # 4. Validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.15, random_state=42, stratify=y_train_all
    )

    # 5. PyTorch tensors
    tr_dataset  = TensorDataset(torch.tensor(X_tr),  torch.tensor(y_tr,  dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long))

    tr_loader  = DataLoader(tr_dataset,  batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    # 6. Model, optimizer, loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    model     = DiseaseNet(input_size, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # 7. Training loop
    best_val_acc = 0.0
    print("\n[INFO] Training started...\n")

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/50 | Loss: {total_loss/len(tr_loader):.4f} | Val Acc: {val_acc*100:.2f}%")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "disease_model.pth")

    # 8. Test evaluation
    model.load_state_dict(torch.load(MODELS_DIR / "disease_model.pth", map_location=device))
    model.eval()

    X_test_t = torch.tensor(X_test_all).to(device)
    with torch.no_grad():
        test_preds = model(X_test_t).argmax(dim=1).cpu().numpy()

    test_acc = accuracy_score(y_test_all, test_preds)
    print(f"\n[OK] Best Val Accuracy : {best_val_acc*100:.2f}%")
    print(f"[OK] Test Accuracy     : {test_acc*100:.2f}%")
    print("\n[INFO] Classification Report (Test Set):")
    print(classification_report(y_test_all, test_preds, target_names=le.classes_))

    # 9. Save metadata
    meta = {
        "feature_cols": feature_cols,
        "label_classes": list(le.classes_),
        "input_size": input_size,
        "num_classes": num_classes,
        "test_accuracy": float(test_acc),
        "val_accuracy": float(best_val_acc),
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    import joblib
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

    print(f"\n[OK] Model saved  → {MODELS_DIR / 'disease_model.pth'}")
    print(f"[OK] Meta saved   → {MODELS_DIR / 'model_meta.json'}")
    print(f"[OK] Encoder saved→ {MODELS_DIR / 'label_encoder.pkl'}")
    print("\n[INFO] Training complete! You can now run: python main.py\n")


if __name__ == "__main__":
    train()
