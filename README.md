# MEDIRAG-XAI
## Explainable Retrieval-Augmented Clinical Decision Support and Patient Education Platform

> **Final Year DLNLP Research Project**  
> A production-ready healthcare AI platform combining Multi-Disease Prediction, Explainable AI (SHAP), Clinical NLP (NER), PDF Report Analysis, Drug Safety Checking, Clinical RAG, and Patient Education Chatbot.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) |
| Mirror | [itachi9604/healthcare-chatbot](https://github.com/itachi9604/healthcare-chatbot) |
| Training Samples | **4,920** |
| Test Samples | **42** |
| Symptom Features | **132 binary features** |
| Disease Classes | **41** |
| Supplementary | Symptom severity, descriptions, precautions |

---

## 🏗️ Project Structure

```
MediRAG-XAI/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── app.py                   # All API routes
│   ├── train_model.py           # Model training (uses real dataset)
│   ├── requirements.txt         # Python dependencies
│   ├── .env                     # Environment variables (edit this)
│   ├── model/
│   │   ├── classifier.py        # PyTorch MLP (41 diseases, 132 features)
│   │   ├── ner.py               # Clinical NER (SciSpacy/spaCy)
│   │   ├── shap_explainer.py    # SHAP KernelExplainer
│   │   ├── report_analyzer.py   # PDF + OCR lab report parser
│   │   ├── drug_checker.py      # Drug safety knowledge base
│   │   └── rag_engine.py        # ChromaDB + SentenceTransformers + Gemini
│   ├── saved_models/
│   │   ├── disease_model.pth    # Trained PyTorch model (after training)
│   │   ├── model_meta.json      # Feature/class metadata
│   │   └── label_encoder.pkl    # Sklearn LabelEncoder
│   ├── data/
│   │   ├── datasets/
│   │   │   ├── Training.csv     # Real Kaggle dataset (4,920 rows)
│   │   │   ├── Testing.csv      # Test set (42 rows)
│   │   │   ├── Symptom_severity.csv
│   │   │   ├── symptom_Description.csv
│   │   │   └── symptom_precaution.csv
│   │   ├── guidelines/          # WHO/CDC clinical guideline documents
│   │   └── patient_docs/        # Patient education documents
│   └── database/
│       └── mongodb.py           # Async MongoDB client (Motor)
├── frontend/
│   ├── index.html               # Landing page
│   ├── doctor.html              # Doctor portal
│   ├── patient.html             # Patient chatbot
│   ├── analytics.html           # Analytics dashboard
│   ├── css/style.css            # Dark medical theme
│   └── js/
│       ├── app.js               # Shared utilities
│       ├── doctor.js            # Doctor portal logic
│       ├── patient.js           # Chatbot logic
│       └── analytics.js         # Chart.js dashboard
├── vector_db/chroma_store/      # ChromaDB persistent storage
├── setup.bat                    # One-click Windows setup
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10 or 3.11
- pip
- (Optional) MongoDB Community Server
- (Optional) Tesseract OCR (for scanned PDF analysis)

---

### Step 1 — Clone / Open Project

Navigate to the project folder:
```powershell
cd "MediRAG-XAI"
```

---

### Step 2 — Create Virtual Environment

```powershell
# Create venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
```

---

### Step 3 — Install Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

---

### Step 4 — Install spaCy Model

```powershell
python -m spacy download en_core_web_sm
```

For SciSpacy (better clinical NER):
```powershell
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

---

### Step 5 — Configure Environment

Edit `backend/.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MONGODB_URI=mongodb://localhost:27017
```

**Getting Gemini API Key:**
1. Go to https://aistudio.google.com/app/apikey
2. Create a new API key
3. Paste into `.env`

> **Note:** The system works without Gemini API key — RAG will return retrieved chunks only, without AI-generated summaries.

---

### Step 6 — Set Up MongoDB (Optional)

1. Download MongoDB Community: https://www.mongodb.com/try/download/community
2. Install and start the MongoDB service
3. The app connects to `mongodb://localhost:27017` by default
4. **Without MongoDB:** The app automatically uses in-memory storage

---

### Step 7 — Train the Disease Prediction Model

```powershell
# From backend/ directory with venv active
python train_model.py
```

**This will:**
- Load `data/datasets/Training.csv` (real Kaggle dataset, 4,920 rows)
- Train a PyTorch MLP neural network (4 layers, 512→256→128→64→41)
- Evaluate on `Testing.csv` (42 test samples)
- Save model to `saved_models/disease_model.pth`
- Save metadata and label encoder

**Expected output:**
```
Training set: 4920 rows × 133 cols
Disease classes: 41
Training started...
  Epoch  1/50 | Loss: 3.7123 | Val Acc: 45.21%
  Epoch  5/50 | Loss: 0.8234 | Val Acc: 89.45%
  ...
  Epoch 50/50 | Loss: 0.0123 | Val Acc: 98.12%

Test Accuracy: 97.56%
Model saved → saved_models/disease_model.pth
```

---

### Step 8 — Start the Backend

```powershell
# From backend/ directory
python main.py
```

Backend runs at: **http://localhost:8000**

API Documentation (Swagger UI): **http://localhost:8000/docs**

---

### Step 9 — Open the Frontend

**Option A — Direct (simple):**
Open `frontend/index.html` in your browser.

**Option B — Local server:**
```powershell
cd frontend
python -m http.server 5500
```
Then open: http://localhost:5500

---

## 🚀 Quick Start (One-Click)

Run the automated setup:
```powershell
setup.bat
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description | Request Body |
|---|---|---|---|
| GET | `/` | API status | — |
| GET | `/health` | Module health check | — |
| POST | `/api/predict` | Disease prediction + SHAP | `{ symptoms: [], patient_age: int, patient_gender: str }` |
| POST | `/api/ner` | Clinical NLP entity extraction | `{ text: str }` |
| POST | `/api/analyze-report` | PDF/image lab report analysis | `multipart/form-data file` |
| POST | `/api/check-drug` | Drug safety check | `{ drug_name, patient_conditions, patient_allergies, is_pregnant, other_drugs }` |
| POST | `/api/chat` | RAG-powered chatbot | `{ message: str, session_id: str }` |
| GET | `/api/analytics` | Dashboard statistics | — |

---

## 🧠 AI Modules

### Module 1 — Multi-Disease Prediction
- **Model:** 4-layer PyTorch MLP (input=132 → 512 → 256 → 128 → 64 → 41 classes)
- **Training:** Adam optimizer, CrossEntropy loss, 50 epochs, BatchNorm + Dropout
- **Dataset:** Real Kaggle Disease-Symptom dataset (4,920 training, 42 test)
- **Output:** Top-3 predictions with confidence % and ICD codes

### Module 2 — Explainable AI (SHAP)
- **Method:** SHAP KernelExplainer
- **Output:** Top-10 feature importances with positive/negative contribution indicators
- **Visualization:** Horizontal bar chart (Chart.js)

### Module 3 — Clinical NLP NER
- **Primary:** SciSpacy (`en_core_sci_sm`) if installed
- **Fallback:** spaCy `en_core_web_sm` + keyword matching + regex
- **Entities:** DISEASE, DRUG, SYMPTOM, MEDICAL_HISTORY, LAB_VALUES

### Module 4 — PDF Report Analyzer
- **Text Extraction:** PyMuPDF (`fitz`)
- **OCR Fallback:** Pillow + pytesseract (for scanned PDFs/images)
- **Extracts:** 20 lab parameters (Glucose, HbA1c, Hemoglobin, TSH, Creatinine, Cholesterol, LDL, HDL, Triglycerides, WBC, RBC, Platelets, SGPT, SGOT, etc.)
- **Normal Ranges:** WHO reference ranges

### Module 5 — Drug Safety Engine
- **Coverage:** 15 common drugs (Metformin, Lisinopril, Aspirin, Atorvastatin, Warfarin, Omeprazole, Insulin, Amoxicillin, Prednisolone, Levothyroxine, Metoprolol, Ciprofloxacin, Furosemide, Ibuprofen, Hydroxychloroquine)
- **Checks:** Contraindications, Pregnancy categories (A-X), Drug-drug interactions, Allergy alerts, Disease interactions

### Module 6 — Clinical RAG
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** ChromaDB (persistent)
- **Generation:** Google Gemini 1.5 Flash
- **Documents:** Diabetes, Hypertension, TB, Heart/Malaria/Dengue guidelines, Drug information, Patient education

### Module 7 — Patient Chatbot
- Powered by the RAG pipeline
- Always shows source citations and evidence chunks
- Session history maintained

### Module 8 — Doctor Portal
- 7 integrated panels in a responsive Bootstrap layout
- Real symptom selection from all 132 Kaggle features

### Module 9 — Analytics Dashboard
- 5 Chart.js visualizations
- Live data from MongoDB or in-memory fallback
- KPI cards with animated counters

---

## 🏥 Supported Diseases (All 41 Kaggle Classes)

Fungal infection, Allergy, GERD, Chronic cholestasis, Drug Reaction, Peptic ulcer, AIDS, Diabetes, Gastroenteritis, Bronchial Asthma, Hypertension, Migraine, Cervical spondylosis, Paralysis, Jaundice, Malaria, Chicken pox, Dengue, Typhoid, Hepatitis A/B/C/D/E, Alcoholic hepatitis, Tuberculosis, Common Cold, Pneumonia, Piles, Heart attack, Varicose veins, Hypothyroidism, Hyperthyroidism, Hypoglycemia, Osteoarthritis, Arthritis, Vertigo, Acne, UTI, Psoriasis, Impetigo

---

## 📈 Model Architecture

```
Input Layer  (132 binary symptom features)
     ↓
Linear(132 → 512) → BatchNorm → ReLU → Dropout(0.3)
     ↓
Linear(512 → 256) → BatchNorm → ReLU → Dropout(0.25)
     ↓
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.2)
     ↓
Linear(128 → 64) → ReLU
     ↓
Linear(64 → 41)  [Output — 41 disease classes]
     ↓
Softmax → Top-3 Predictions
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` with venv active |
| `Model file not found` | Run `python train_model.py` first |
| `ChromaDB error` | Check `vector_db/chroma_store/` directory exists |
| `Gemini API error` | Set `GEMINI_API_KEY` in `.env` |
| `MongoDB connection refused` | Start MongoDB service or ignore (in-memory fallback works) |
| `spaCy model not found` | Run `python -m spacy download en_core_web_sm` |
| `CORS error in browser` | Ensure FastAPI is running at localhost:8000 |

---

## ⚠️ Disclaimer

This platform is developed for **academic research and educational purposes only**. It is NOT intended for:
- Clinical diagnosis or treatment decisions
- Prescription of medications
- Replacement of professional medical advice

Always consult qualified healthcare professionals for medical decisions.

---

## 📚 References

1. Kaggle Disease Symptom Dataset — https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
2. WHO Clinical Guidelines — https://www.who.int/publications
3. CDC Treatment Guidelines — https://www.cdc.gov
4. American Diabetes Association Standards 2023
5. ESC/ESH Hypertension Guidelines 2023
6. SHAP: A Unified Approach to Interpreting Model Predictions (Lundberg & Lee, 2017)
7. ChromaDB Documentation — https://docs.trychroma.com
8. SentenceTransformers — https://www.sbert.net
9. Google Gemini API — https://ai.google.dev

---

*MEDIRAG-XAI — Final Year DLNLP Project | Built with FastAPI, PyTorch, ChromaDB, Gemini, Bootstrap 5, Chart.js*
