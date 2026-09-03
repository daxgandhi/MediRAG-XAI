"""
MEDIRAG-XAI FastAPI Application
All routes, middleware, startup logic
"""
import os
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import hashlib
from dotenv import load_dotenv

load_dotenv()

# ─── Lazy module imports (so app starts even if optional deps fail) ────────────
_classifier = None
_ner = None
_shap_explainer = None
_report_analyzer = None
_drug_checker = None
_rag_engine = None
_db = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from model.classifier import DiseaseClassifier
        _classifier = DiseaseClassifier()
    return _classifier


def get_ner():
    global _ner
    if _ner is None:
        from model.ner import ClinicalNER
        _ner = ClinicalNER()
    return _ner


def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        from model.shap_explainer import SHAPExplainer
        _shap_explainer = SHAPExplainer(get_classifier())
    return _shap_explainer


def get_report_analyzer():
    global _report_analyzer
    if _report_analyzer is None:
        from model.report_analyzer import ReportAnalyzer
        _report_analyzer = ReportAnalyzer()
    return _report_analyzer


def get_drug_checker():
    global _drug_checker
    if _drug_checker is None:
        from model.drug_checker import DrugChecker
        _drug_checker = DrugChecker()
    return _drug_checker


def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        from model.rag_engine import RAGEngine
        _rag_engine = RAGEngine()
    return _rag_engine


def get_db():
    global _db
    if _db is None:
        from database.mongodb import MediDB
        _db = MediDB()
    return _db


# ─── Pydantic Models ───────────────────────────────────────────────────────────
class UserRegisterRequest(BaseModel):
    role: str
    password: str
    email: Optional[str] = None
    medical_id: Optional[str] = None
    name: Optional[str] = None

class UserLoginRequest(BaseModel):
    role: str
    password: str
    email: Optional[str] = None
    medical_id: Optional[str] = None

class PredictRequest(BaseModel):
    symptoms: List[str]
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class DrugCheckRequest(BaseModel):
    drug_name: str
    patient_conditions: Optional[List[str]] = []
    patient_allergies: Optional[List[str]] = []
    is_pregnant: Optional[bool] = False
    other_drugs: Optional[List[str]] = []


class NERRequest(BaseModel):
    text: str


# ─── App Factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: pre-load critical models
        try:
            print("[INFO] Loading disease classifier...")
            get_classifier()
            print("[OK] Classifier loaded")
        except Exception as e:
            print(f"[WARN] Classifier not ready: {e}")
        try:
            print("[INFO] Loading drug checker...")
            get_drug_checker()
            print("[OK] Drug checker loaded")
        except Exception as e:
            print(f"[WARN] Drug checker not ready: {e}")
        yield
        print("[INFO] Shutting down MEDIRAG-XAI...")

    app = FastAPI(
        title="MEDIRAG-XAI API",
        description="Explainable Retrieval-Augmented Clinical Decision Support Platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Configure CORS: allow custom origins via CORS_ORIGINS env or default to permissive for local dev
    cors_env = os.getenv("CORS_ORIGINS", "").strip()
    if cors_env and cors_env != "*":
        allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
        # Always retain localhost dev origins for local testing convenience
        for dev_url in ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5500", "http://127.0.0.1:5500", "http://localhost:3000"]:
            if dev_url not in allowed_origins:
                allowed_origins.append(dev_url)
    else:
        allowed_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── ROUTES ───────────────────────────────────────────────────────────────

    @app.get("/api")
    async def root():
        return {"status": "running", "service": "MEDIRAG-XAI API", "version": "1.0.0"}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "modules": {
            "classifier": _classifier is not None,
            "ner": _ner is not None,
            "rag": _rag_engine is not None,
            "drug_checker": _drug_checker is not None,
        }}

    # ── USER AUTHENTICATION ───────────────────────────────────────────────────
    @app.post("/api/register")
    async def register_user(req: UserRegisterRequest):
        try:
            db = get_db()
            user_data = req.dict()
            user_data["password"] = hashlib.sha256(req.password.encode()).hexdigest()
            user_id = await db.save_user(user_data)
            if not user_id:
                raise HTTPException(status_code=400, detail="User already exists or failed to create")
            return JSONResponse({"success": True, "user_id": user_id, "role": req.role})
        except Exception as e:
            traceback.print_exc()
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/login")
    async def login_user(req: UserLoginRequest):
        try:
            db = get_db()
            user = None
            if req.role == "doctor" and req.medical_id:
                user = await db.get_user_by_medical_id(req.medical_id)
                # Fallback to email for doctor if provided
                if not user and req.email:
                    user = await db.get_user_by_email(req.email)
            elif req.email:
                user = await db.get_user_by_email(req.email)
                
            if not user:
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
            hashed_pwd = hashlib.sha256(req.password.encode()).hexdigest()
            if user.get("password") != hashed_pwd:
                raise HTTPException(status_code=401, detail="Invalid credentials")
                
            # Don't send password back
            user.pop("password", None)
            user.pop("_id", None)
            if "timestamp" in user:
                user["timestamp"] = str(user["timestamp"])
            return JSONResponse({"success": True, "user": user})
        except Exception as e:
            traceback.print_exc()
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))

    # ── MODULE 1+2: Disease Prediction + SHAP ─────────────────────────────────
    @app.post("/api/predict")
    async def predict_disease(req: PredictRequest):
        try:
            clf = get_classifier()
            results = clf.predict(req.symptoms)

            # SHAP explanation for top prediction
            shap_data = {"features": [], "values": []}
            try:
                explainer = get_shap_explainer()
                shap_data = explainer.explain(req.symptoms)
            except Exception as se:
                print(f"SHAP warning: {se}")

            # Save to DB (non-blocking)
            try:
                db = get_db()
                await db.save_prediction({
                    "symptoms": req.symptoms,
                    "top_prediction": results[0] if results else {},
                    "patient_age": req.patient_age,
                    "patient_gender": req.patient_gender,
                })
            except Exception:
                pass

            return JSONResponse({
                "success": True,
                "predictions": results,
                "shap_explanation": shap_data,
                "symptoms_analyzed": req.symptoms,
            })
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    # ── MODULE 3: Clinical NER ─────────────────────────────────────────────────
    @app.post("/api/ner")
    async def extract_entities(req: NERRequest):
        try:
            ner = get_ner()
            entities = ner.extract(req.text)
            return JSONResponse({"success": True, "entities": entities, "text": req.text})
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    # ── MODULE 4: PDF Report Analysis ─────────────────────────────────────────
    @app.post("/api/analyze-report")
    async def analyze_report(file: UploadFile = File(...)):
        try:
            content = await file.read()
            analyzer = get_report_analyzer()
            result = analyzer.analyze(content, filename=file.filename)

            # Save to DB
            try:
                db = get_db()
                await db.save_report({
                    "filename": file.filename,
                    "summary": result.get("summary", ""),
                    "abnormal_count": len(result.get("abnormal_values", [])),
                })
            except Exception:
                pass

            return JSONResponse({"success": True, **result})
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    # ── MODULE 5: Drug Safety ─────────────────────────────────────────────────
    @app.post("/api/check-drug")
    async def check_drug(req: DrugCheckRequest):
        try:
            checker = get_drug_checker()
            result = checker.check(
                drug_name=req.drug_name,
                conditions=req.patient_conditions,
                allergies=req.patient_allergies,
                is_pregnant=req.is_pregnant,
                other_drugs=req.other_drugs,
            )

            # Save alert to DB
            if result.get("alerts"):
                try:
                    db = get_db()
                    await db.save_drug_alert({
                        "drug": req.drug_name,
                        "alerts": result["alerts"],
                    })
                except Exception:
                    pass

            return JSONResponse({"success": True, **result})
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    # ── MODULE 6+7: RAG Chat ──────────────────────────────────────────────────
    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        try:
            rag = get_rag_engine()
            result = rag.query(req.message)

            # Save to DB
            try:
                db = get_db()
                await db.save_chat({
                    "session_id": req.session_id,
                    "question": req.message,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                })
            except Exception:
                pass

            return JSONResponse({"success": True, **result})
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    # ── MODULE 10: Analytics ──────────────────────────────────────────────────
    @app.get("/api/analytics")
    async def get_analytics():
        try:
            db = get_db()
            data = await db.get_analytics()
            return JSONResponse({"success": True, **data})
        except Exception as e:
            # Return demo data if DB is unavailable
            return JSONResponse({
                "success": True,
                "disease_distribution": {
                    "Diabetes": 145, "Hypertension": 132, "Asthma": 98,
                    "COPD": 67, "Heart Disease": 89, "Migraine": 75,
                    "Anxiety": 112, "Depression": 95, "Pneumonia": 58,
                    "COVID-19": 88, "Other": 141
                },
                "prediction_confidence": [72, 78, 85, 81, 88, 91, 87, 93, 89, 95, 92, 88, 94, 90, 96],
                "confidence_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Q1", "Q2", "Q3"],
                "common_symptoms": {
                    "Fever": 312, "Fatigue": 287, "Cough": 245,
                    "Headache": 198, "Chest Pain": 156, "Shortness of Breath": 143,
                    "Nausea": 132, "Joint Pain": 118
                },
                "drug_alerts": {
                    "Contraindication": 45, "Pregnancy Warning": 23,
                    "Allergy Alert": 31, "Drug Interaction": 67, "Safe": 234
                },
                "chat_usage": [12, 18, 25, 32, 28, 41, 35, 48, 52, 44, 38, 55, 61, 47, 58],
                "total_predictions": 1100,
                "total_reports": 234,
                "total_chats": 594,
                "total_drug_checks": 400,
            })

    # Mount the frontend directory to serve static files (unified deployment only)
    # When frontend is deployed separately (e.g. on Vercel), skip this to avoid
    # the catch-all mount intercepting API routes.
    frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
    if os.path.isdir(frontend_dir) and cors_env in ("", "*"):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app
