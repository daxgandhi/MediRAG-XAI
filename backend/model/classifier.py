"""
Disease Prediction Neural Network Classifier
Trained on REAL Kaggle Dataset: 132 symptoms, 41 diseases, 4920 rows
"""
import json
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "saved_models" / "disease_model.pth"
META_PATH  = BASE_DIR / "saved_models" / "model_meta.json"
ENC_PATH   = BASE_DIR / "saved_models" / "label_encoder.pkl"

# Severity mapping (for all 41 Kaggle disease classes)
SEVERITY_MAP = {
    "Fungal infection": "Low", "Allergy": "Low", "GERD": "Low",
    "Chronic cholestasis": "Moderate", "Drug Reaction": "Moderate",
    "Peptic ulcer diseae": "Moderate", "AIDS": "Critical",
    "Diabetes ": "Moderate", "Gastroenteritis": "Moderate",
    "Bronchial Asthma": "Moderate", "Hypertension ": "High",
    "Migraine": "Moderate", "Cervical spondylosis": "Low",
    "Paralysis (brain hemorrhage)": "Critical", "Jaundice": "High",
    "Malaria": "High", "Chicken pox": "Low", "Dengue": "High",
    "Typhoid": "Moderate", "hepatitis A": "Moderate",
    "Hepatitis B": "High", "Hepatitis C": "High",
    "Hepatitis D": "High", "Hepatitis E": "Moderate",
    "Alcoholic hepatitis": "High", "Tuberculosis": "High",
    "Common Cold": "Low", "Pneumonia": "High",
    "Dimorphic hemmorhoids(piles)": "Moderate", "Heart attack": "Critical",
    "Varicose veins": "Low", "Hypothyroidism": "Low",
    "Hyperthyroidism": "Moderate", "Hypoglycemia": "High",
    "Osteoarthristis": "Low", "Arthritis": "Low",
    "(vertigo) Paroymsal  Positional Vertigo": "Low",
    "Acne": "Low", "Urinary tract infection": "Low",
    "Psoriasis": "Low", "Impetigo": "Low",
}

ICD_MAP = {
    "Fungal infection": "B49", "Allergy": "T78.4", "GERD": "K21",
    "Chronic cholestasis": "K83.1", "Drug Reaction": "T88.7",
    "Peptic ulcer diseae": "K27", "AIDS": "B24",
    "Diabetes ": "E11", "Gastroenteritis": "A09",
    "Bronchial Asthma": "J45", "Hypertension ": "I10",
    "Migraine": "G43", "Cervical spondylosis": "M47.8",
    "Paralysis (brain hemorrhage)": "G81", "Jaundice": "R17",
    "Malaria": "B50", "Chicken pox": "B01", "Dengue": "A90",
    "Typhoid": "A01", "hepatitis A": "B15", "Hepatitis B": "B16",
    "Hepatitis C": "B17.1", "Hepatitis D": "B17.0", "Hepatitis E": "B17.2",
    "Alcoholic hepatitis": "K70.1", "Tuberculosis": "A15",
    "Common Cold": "J00", "Pneumonia": "J18",
    "Dimorphic hemmorhoids(piles)": "K64", "Heart attack": "I21",
    "Varicose veins": "I83", "Hypothyroidism": "E03",
    "Hyperthyroidism": "E05", "Hypoglycemia": "E16.0",
    "Osteoarthristis": "M19", "Arthritis": "M13",
    "(vertigo) Paroymsal  Positional Vertigo": "H81.1",
    "Acne": "L70", "Urinary tract infection": "N39",
    "Psoriasis": "L40", "Impetigo": "L01",
}


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


class DiseaseClassifier:
    def __init__(self):
        self.device = torch.device("cpu")
        self.feature_cols = []
        self.label_classes = []
        self.symptom_to_idx = {}
        self.model = None

        # Self-healing: if model weights or metadata are missing, train on startup
        if not META_PATH.exists() or not MODEL_PATH.exists() or not ENC_PATH.exists():
            print("[INFO] Model weights or metadata not found. Initializing auto-training...")
            try:
                from train_model import train
                train(epochs=30)
            except Exception as e:
                print(f"[WARN] Auto-training error: {e}")

        if META_PATH.exists():
            with open(META_PATH) as f:
                meta = json.load(f)
            self.feature_cols   = meta["feature_cols"]
            self.label_classes  = meta["label_classes"]
            self.symptom_to_idx = {s: i for i, s in enumerate(self.feature_cols)}
            input_size  = meta["input_size"]
            num_classes = meta["num_classes"]
            self.model = DiseaseNet(input_size, num_classes).to(self.device)

            if MODEL_PATH.exists():
                try:
                    state = torch.load(MODEL_PATH, map_location=self.device)
                    self.model.load_state_dict(state)
                    self.model.eval()
                    print(f"[OK] Model loaded — {input_size} features, {num_classes} diseases")
                except Exception as e:
                    print(f"[WARN] Could not load weights: {e}")
                    self.model.eval()
            else:
                self.model.eval()

            if ENC_PATH.exists():
                self.label_encoder = joblib.load(ENC_PATH)
            else:
                self.label_encoder = None
        else:
            print("[WARN] Running with fallback dataset metadata")
            self._init_fallback()

    def _init_fallback(self):
        """Minimal fallback when model hasn't been trained yet."""
        from pathlib import Path
        import pandas as pd
        data_path = BASE_DIR / "data" / "datasets" / "Training.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            self.feature_cols  = [c for c in df.columns if c != "prognosis"]
            self.label_classes = sorted(df["prognosis"].unique().tolist())
        else:
            self.feature_cols  = []
            self.label_classes = []
        self.symptom_to_idx = {s: i for i, s in enumerate(self.feature_cols)}
        input_size  = len(self.feature_cols) or 132
        num_classes = len(self.label_classes) or 41
        self.model = DiseaseNet(input_size, num_classes).to(self.device)
        self.model.eval()
        self.label_encoder = None

    def _symptoms_to_vector(self, symptom_list: list) -> np.ndarray:
        vec = np.zeros(len(self.feature_cols), dtype=np.float32)
        for sym in symptom_list:
            normalized = sym.lower().strip().replace(" ", "_").replace("-", "_")
            if normalized in self.symptom_to_idx:
                vec[self.symptom_to_idx[normalized]] = 1.0
            else:
                # Partial / fuzzy match
                for known in self.symptom_to_idx:
                    if normalized in known or known in normalized:
                        vec[self.symptom_to_idx[known]] = 1.0
                        break
        return vec

    def predict(self, symptoms: list, top_k: int = 3) -> list:
        if not self.feature_cols:
            return [{"disease": "Model not trained", "confidence": 0, "severity": "N/A", "icd_code": "N/A"}]

        vec = self._symptoms_to_vector(symptoms)
        x   = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        results = []
        for idx in top_indices:
            disease    = self.label_classes[idx]
            confidence = float(probs[idx]) * 100
            results.append({
                "disease":    disease.strip(),
                "confidence": round(confidence, 1),
                "severity":   SEVERITY_MAP.get(disease, "Moderate"),
                "icd_code":   ICD_MAP.get(disease, "Z99"),
            })
        return results

    def predict_vector(self, symptom_vector: np.ndarray) -> np.ndarray:
        """For SHAP: accepts raw feature vector, returns probability array."""
        x = torch.tensor(symptom_vector, dtype=torch.float32).to(self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def symptoms_to_vector_np(self, symptom_list: list) -> np.ndarray:
        return self._symptoms_to_vector(symptom_list)

    @property
    def symptoms(self):
        return self.feature_cols

    @property
    def diseases(self):
        return self.label_classes
