"""
Quick verification script — runs before full server startup
Checks that all modules are importable and dataset exists.
Run: python verify.py
"""
import sys
import os
from pathlib import Path

BASE = Path(__file__).parent
errors = []
warnings = []

print("\n" + "="*55)
print("  MEDIRAG-XAI System Verification")
print("="*55 + "\n")

# 1. Dataset
train_csv = BASE / "data" / "datasets" / "Training.csv"
test_csv  = BASE / "data" / "datasets" / "Testing.csv"
if train_csv.exists():
    import pandas as pd
    df = pd.read_csv(train_csv)
    print(f"  [OK] Training.csv   -> {len(df)} rows, {df.shape[1]} cols")
else:
    errors.append("Training.csv not found — dataset missing")
    print(f"  [ERROR] Training.csv   -> NOT FOUND")

if test_csv.exists():
    print(f"  [OK] Testing.csv    -> found")
else:
    warnings.append("Testing.csv not found")
    print(f"  [WARN]  Testing.csv   -> not found")

# 2. Model
model_path = BASE / "saved_models" / "disease_model.pth"
meta_path  = BASE / "saved_models" / "model_meta.json"
if model_path.exists() and meta_path.exists():
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  [OK] Trained model  -> {meta['num_classes']} classes, {meta['input_size']} features")
    print(f"     Test accuracy  -> {meta.get('test_accuracy', '?'):.2%}")
else:
    warnings.append("Model not trained yet — run: python train_model.py")
    print(f"  [WARN]  Model file    -> NOT FOUND (run train_model.py)")

# 3. Guidelines
guidelines = list((BASE / "data" / "guidelines").glob("*.md"))
patient_docs = list((BASE / "data" / "patient_docs").glob("*.md"))
print(f"  [OK] Guidelines     -> {len(guidelines)} documents")
print(f"  [OK] Patient docs   -> {len(patient_docs)} documents")

# 4. FastAPI
try:
    import fastapi
    print(f"  [OK] FastAPI        -> v{fastapi.__version__}")
except ImportError as e:
    errors.append(f"FastAPI not installed: {e}")
    print(f"  [ERROR] FastAPI        -> NOT INSTALLED")

# 5. PyTorch
try:
    import torch
    print(f"  [OK] PyTorch        -> v{torch.__version__}")
except ImportError:
    warnings.append("PyTorch not installed — run: pip install torch")
    print(f"  [WARN]  PyTorch       -> not installed")

# 6. spaCy
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        print(f"  [OK] spaCy          -> en_core_web_sm loaded")
    except OSError:
        try:
            nlp = spacy.load("en_core_sci_sm")
            print(f"  [OK] spaCy          -> en_core_sci_sm (SciSpacy) loaded")
        except OSError:
            warnings.append("No spaCy model installed — run: python -m spacy download en_core_web_sm")
            print(f"  [WARN]  spaCy model   -> not downloaded")
except ImportError:
    warnings.append("spaCy not installed")
    print(f"  [WARN]  spaCy         -> not installed")

# 7. ChromaDB
try:
    import chromadb
    print(f"  [OK] ChromaDB       -> v{chromadb.__version__}")
except ImportError:
    warnings.append("ChromaDB not installed — RAG engine unavailable")
    print(f"  [WARN]  ChromaDB      -> not installed")

# 8. Sentence Transformers
try:
    import sentence_transformers
    print(f"  [OK] SentenceTransformers -> v{sentence_transformers.__version__}")
except ImportError:
    warnings.append("sentence-transformers not installed — RAG engine unavailable")
    print(f"  [WARN]  SentenceTransformers -> not installed")

# 9. LLM Provider & Keys
try:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
except ImportError:
    pass

provider = os.getenv("LLM_PROVIDER", "gemini").lower()
print(f"  [INFO]  LLM Provider  -> {provider}")

if provider == "grok":
    xai_key = os.getenv("XAI_API_KEY", "")
    if xai_key and xai_key != "your_xai_api_key_here":
        print(f"  [OK] xAI API key    -> configured (Model: {os.getenv('XAI_MODEL', 'grok-2')})")
    else:
        warnings.append("XAI_API_KEY not set — RAG will return evidence only (no AI summary)")
        print(f"  [WARN]  xAI API key   -> NOT SET (edit backend/.env)")
else:
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        print(f"  [OK] Gemini API key -> configured (Model: {os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')})")
    else:
        warnings.append("GEMINI_API_KEY not set — RAG will return evidence only (no AI summary)")
        print(f"  [WARN]  Gemini API key -> NOT SET (edit backend/.env)")

# 10. MongoDB
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
print(f"  [INFO]  MongoDB URI   -> {mongo_uri} (optional — in-memory fallback if unavailable)")

# 11. PyMuPDF
try:
    import fitz
    print(f"  [OK] PyMuPDF        -> v{fitz.version[0]}")
except ImportError:
    warnings.append("PyMuPDF not installed — PDF text extraction unavailable")
    print(f"  [WARN]  PyMuPDF       -> not installed")

# 12. SHAP
try:
    import shap
    print(f"  [OK] SHAP           -> v{shap.__version__}")
except ImportError:
    warnings.append("SHAP not installed — XAI fallback will be used")
    print(f"  [WARN]  SHAP          -> not installed")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*55)
if errors:
    print(f"  [ERROR] ERRORS ({len(errors)}) — must fix before running:")
    for e in errors:
        print(f"     • {e}")
if warnings:
    print(f"  [WARN]  WARNINGS ({len(warnings)}) — optional but recommended:")
    for w in warnings:
        print(f"     • {w}")
if not errors and not warnings:
    print("  [OK] All checks passed! Ready to run: python main.py")
elif not errors:
    print(f"\n  [OK] No critical errors. Start server: python main.py")
    print(f"  (Warnings above are non-critical)")
else:
    print(f"\n  Please fix errors above before starting the server.")
print("="*55 + "\n")

if errors:
    sys.exit(1)
