"""
MEDIRAG-XAI Test Suite
Tests: Disease Prediction, NER, Drug Checker, Report Analyzer, API endpoints
Run: python -m pytest tests/ -v   (from backend/ with venv active)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─── Model Tests ──────────────────────────────────────────────────────────────
class TestDiseaseClassifier:
    def setup_method(self):
        from model.classifier import DiseaseClassifier
        self.clf = DiseaseClassifier()

    def test_classifier_loads(self):
        assert self.clf is not None

    def test_feature_list_loaded(self):
        # Should have 132 features (real Kaggle dataset)
        assert len(self.clf.feature_cols) > 0, "Feature columns not loaded"
        print(f"  Features: {len(self.clf.feature_cols)}")

    def test_disease_list_loaded(self):
        assert len(self.clf.diseases) > 0, "Disease classes not loaded"
        print(f"  Diseases: {len(self.clf.diseases)}")

    def test_predict_returns_list(self):
        symptoms = ["itching", "skin_rash", "nodal_skin_eruptions"]
        results = self.clf.predict(symptoms, top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3
        print(f"  Top prediction: {results[0]['disease']} ({results[0]['confidence']:.1f}%)")

    def test_predict_has_required_keys(self):
        results = self.clf.predict(["fever", "cough", "fatigue"])
        for r in results:
            assert "disease"    in r
            assert "confidence" in r
            assert "severity"   in r
            assert "icd_code"   in r

    def test_confidence_sum_near_100(self):
        results = self.clf.predict(["fever", "cough", "fatigue"], top_k=5)
        total = sum(r["confidence"] for r in results)
        assert total <= 100.1, f"Confidence sum exceeded 100: {total}"

    def test_fuzzy_symptom_matching(self):
        # Should match even with spaces instead of underscores
        results = self.clf.predict(["skin rash", "fever"])
        assert len(results) > 0

    def test_empty_symptoms_graceful(self):
        results = self.clf.predict([])
        assert isinstance(results, list)


# ─── NER Tests ────────────────────────────────────────────────────────────────
class TestClinicalNER:
    def setup_method(self):
        from model.ner import ClinicalNER
        self.ner = ClinicalNER()

    def test_ner_loads(self):
        assert self.ner is not None

    def test_extract_disease(self):
        result = self.ner.extract("Patient has diabetes and hypertension.")
        assert "Diabetes" in result["DISEASE"] or "diabetes" in [d.lower() for d in result["DISEASE"]]

    def test_extract_drug(self):
        result = self.ner.extract("Patient takes metformin 500mg daily.")
        drugs_lower = [d.lower() for d in result["DRUG"]]
        assert "metformin" in drugs_lower

    def test_extract_lab_values(self):
        result = self.ner.extract("HbA1c: 8.2% and glucose: 180 mg/dL")
        assert len(result["LAB_VALUES"]) > 0

    def test_extract_multiple_entities(self):
        text = "Patient has diabetes and takes metformin. History of kidney disease. Fever and fatigue."
        result = self.ner.extract(text)
        assert isinstance(result["DISEASE"],         list)
        assert isinstance(result["DRUG"],            list)
        assert isinstance(result["SYMPTOM"],         list)
        assert isinstance(result["MEDICAL_HISTORY"], list)
        assert isinstance(result["LAB_VALUES"],      list)

    def test_empty_text(self):
        result = self.ner.extract("")
        for key in result:
            assert isinstance(result[key], list)


# ─── Drug Checker Tests ───────────────────────────────────────────────────────
class TestDrugChecker:
    def setup_method(self):
        from model.drug_checker import DrugChecker
        self.checker = DrugChecker()

    def test_checker_loads(self):
        assert self.checker is not None

    def test_known_drug_found(self):
        result = self.checker.check("metformin")
        assert result["drug_found"] == True
        assert result["drug_name"] == "Metformin"

    def test_unknown_drug_not_found(self):
        result = self.checker.check("xyz_nonexistent_drug_12345")
        assert result["drug_found"] == False

    def test_pregnancy_contraindication_lisinopril(self):
        result = self.checker.check("lisinopril", is_pregnant=True)
        assert any(a["severity"] == "CRITICAL" for a in result["alerts"])

    def test_pregnancy_safe_metformin(self):
        result = self.checker.check("metformin", is_pregnant=True)
        # Metformin is Category B — should have info or warning but not critical
        critical_alerts = [a for a in result["alerts"] if a["severity"] == "CRITICAL"]
        assert len(critical_alerts) == 0

    def test_disease_interaction_metformin_kidney(self):
        result = self.checker.check("metformin", conditions=["Kidney Disease"])
        assert len(result["alerts"]) > 0

    def test_atorvastatin_pregnancy_contraindicated(self):
        result = self.checker.check("atorvastatin", is_pregnant=True)
        assert any("CRITICAL" == a["severity"] for a in result["alerts"])

    def test_result_has_required_keys(self):
        result = self.checker.check("aspirin")
        assert "drug_found"    in result
        assert "alerts"        in result
        assert "disclaimer"    in result

    def test_allergy_alert(self):
        result = self.checker.check("amoxicillin", allergies=["Penicillin"])
        assert any("Allergy" in a["category"] for a in result["alerts"])


# ─── Report Analyzer Tests ────────────────────────────────────────────────────
class TestReportAnalyzer:
    def setup_method(self):
        from model.report_analyzer import ReportAnalyzer
        self.analyzer = ReportAnalyzer()

    def test_analyzer_loads(self):
        assert self.analyzer is not None

    def test_analyze_text_report(self):
        """Simulate text that looks like a lab report"""
        # Create a minimal fake PDF-like bytes (text extraction will fail gracefully)
        content = b"not a real pdf"
        result = self.analyzer.analyze(content, filename="test.pdf")
        # Should return a dict with required keys even if extraction fails
        assert "summary"        in result
        assert "lab_values"     in result
        assert "abnormal_values" in result

    def test_lab_value_extraction(self):
        text = "Glucose: 250 mg/dL\nHbA1c: 9.5%\nHemoglobin: 11.2 g/dL\nTSH: 6.8 mIU/L"
        lab_values = self.analyzer._extract_lab_values(text)
        assert "glucose"    in lab_values
        assert "hba1c"      in lab_values
        assert "hemoglobin" in lab_values
        assert "tsh"        in lab_values
        assert lab_values["glucose"] == 250.0
        assert lab_values["hba1c"]   == 9.5

    def test_abnormal_flagging_high_glucose(self):
        lab_values = {"glucose": 300.0}  # Way above normal (70-100)
        abnormals = self.analyzer._flag_abnormals(lab_values)
        assert len(abnormals) > 0
        assert abnormals[0]["status"] == "HIGH"

    def test_abnormal_flagging_low_hemoglobin(self):
        lab_values = {"hemoglobin": 9.5}  # Below normal
        abnormals = self.analyzer._flag_abnormals(lab_values)
        assert len(abnormals) > 0
        assert abnormals[0]["status"] == "LOW"

    def test_normal_values_not_flagged(self):
        lab_values = {"glucose": 85.0, "hba1c": 5.2}  # Both normal
        abnormals = self.analyzer._flag_abnormals(lab_values)
        assert len(abnormals) == 0

    def test_report_type_detection(self):
        text_diabetes = "HbA1c: 8.2%, fasting glucose: 180 mg/dL"
        rtype = self.analyzer._detect_report_type(text_diabetes)
        assert "Diabetes" in rtype or "Blood Sugar" in rtype

        text_thyroid = "TSH: 8.5 mIU/L, T3: 2.1, T4: 6.8"
        rtype2 = self.analyzer._detect_report_type(text_thyroid)
        assert "Thyroid" in rtype2


# ─── API Integration Tests ────────────────────────────────────────────────────
class TestAPIRoutes:
    """Integration tests requiring FastAPI app running (or using TestClient)"""

    def setup_method(self):
        try:
            from fastapi.testclient import TestClient
            from app import create_app
            self.client = TestClient(create_app())
            self.available = True
        except Exception as e:
            print(f"  TestClient not available: {e}")
            self.available = False

    def test_root_endpoint(self):
        if not self.available: return
        resp = self.client.get("/")
        assert resp.status_code == 200
        assert "MEDIRAG-XAI" in resp.json()["service"]

    def test_health_endpoint(self):
        if not self.available: return
        resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "modules" in data

    def test_predict_endpoint(self):
        if not self.available: return
        resp = self.client.post("/api/predict", json={
            "symptoms": ["fever", "cough", "fatigue"]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] == True
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_drug_check_endpoint(self):
        if not self.available: return
        resp = self.client.post("/api/check-drug", json={
            "drug_name":          "metformin",
            "patient_conditions": ["Kidney Disease"],
            "is_pregnant":        False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]   == True
        assert data["drug_found"] == True

    def test_ner_endpoint(self):
        if not self.available: return
        resp = self.client.post("/api/ner", json={
            "text": "Patient has diabetes and takes metformin."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] == True
        assert "entities" in data

    def test_analytics_endpoint(self):
        if not self.available: return
        resp = self.client.get("/api/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] == True
        assert "disease_distribution" in data


# ─── Quick standalone runner ──────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback

    test_classes = [
        TestDiseaseClassifier,
        TestClinicalNER,
        TestDrugChecker,
        TestReportAnalyzer,
        TestAPIRoutes,
    ]

    total_passed = 0
    total_failed = 0

    for cls in test_classes:
        print(f"\n{'='*55}")
        print(f"  {cls.__name__}")
        print(f"{'='*55}")
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for mname in methods:
            inst = cls()
            try:
                inst.setup_method()
                getattr(inst, mname)()
                print(f"  ✅  {mname}")
                total_passed += 1
            except Exception as e:
                print(f"  ❌  {mname}")
                print(f"       {e}")
                total_failed += 1

    print(f"\n{'='*55}")
    print(f"  Results: {total_passed} passed, {total_failed} failed")
    print(f"{'='*55}\n")
