"""
PDF & Image Medical Report Analyzer
Uses PyMuPDF for text extraction, pytesseract/Pillow for OCR fallback
Extracts: lab values, risk indicators, abnormal flags
"""
import re
import io
from typing import Dict, Any, List

# Normal ranges for common lab tests
NORMAL_RANGES = {
    "glucose":        (70,  100,  "mg/dL",  "Fasting Blood Glucose"),
    "hba1c":          (4.0, 5.7,  "%",      "HbA1c"),
    "hemoglobin_m":   (13.5, 17.5, "g/dL",  "Hemoglobin (Male)"),
    "hemoglobin_f":   (12.0, 15.5, "g/dL",  "Hemoglobin (Female)"),
    "hemoglobin":     (12.0, 17.5, "g/dL",  "Hemoglobin"),
    "pcv":            (35.0, 60.0, "%",     "PCV (Hematocrit)"),
    "mcv":            (80.0, 99.0, "fL",    "MCV"),
    "mch":            (27.0, 31.0, "pg",    "MCH"),
    "mchc":           (32.0, 37.0, "%",     "MCHC"),
    "rdw":            (11.0, 17.0, "%",     "RDW"),
    "neutrophils":    (40.0, 70.0, "%",     "Neutrophils"),
    "lymphocytes":    (20.0, 45.0, "%",     "Lymphocytes"),
    "eosinophils":    (0.0,  6.0,  "%",     "Eosinophils"),
    "monocytes":      (0.0,  8.0,  "%",     "Monocytes"),
    "basophils":      (0.0,  1.0,  "%",     "Basophils"),
    "tsh":            (0.4,  4.0,  "mIU/L", "TSH"),
    "t3":             (0.8,  2.0,  "ng/mL", "Total T3"),
    "t4":             (5.0,  12.0, "µg/dL", "Total T4"),
    "ft3":            (2.0,  4.4,  "pg/mL", "Free T3 (FT3)"),
    "ft4":            (0.8,  1.8,  "ng/dL", "Free T4 (FT4)"),
    "creatinine_m":   (0.74, 1.35, "mg/dL", "Creatinine (Male)"),
    "creatinine_f":   (0.59, 1.04, "mg/dL", "Creatinine (Female)"),
    "creatinine":     (0.59, 1.35, "mg/dL", "Creatinine"),
    "cholesterol":    (0,    200,  "mg/dL", "Total Cholesterol"),
    "ldl":            (0,    100,  "mg/dL", "LDL Cholesterol"),
    "hdl":            (40,   60,   "mg/dL", "HDL Cholesterol"),
    "triglycerides":  (0,    150,  "mg/dL", "Triglycerides"),
    "wbc":            (4000, 11000,"cells/µL","WBC"),
    "rbc":            (4.0,  6.0,  "M/µL",  "RBC"),
    "platelets":      (150000, 450000, "/µL","Platelets"),
    "sgpt":           (7,    56,   "U/L",   "SGPT (ALT)"),
    "sgot":           (5,    40,   "U/L",   "SGOT (AST)"),
    "uric_acid":      (2.6,  7.2,  "mg/dL", "Uric Acid"),
    "sodium":         (135,  145,  "mEq/L", "Sodium"),
    "potassium":      (3.5,  5.0,  "mEq/L", "Potassium"),
    "bilirubin":      (0.1,  1.2,  "mg/dL", "Total Bilirubin"),
}

EXTRACT_PATTERNS = [
    (r"(?:fasting\s+)?(?:blood\s+)?glucose[\s\n\-:]*(\d+\.?\d*)", "glucose"),
    (r"HbA1c[\s\n\-:]*(\d+\.?\d*)", "hba1c"),
    (r"(?:haemoglobin|hemoglobin|hb)(?:\s*\([^\)]*\))?[\s\n\-:=]*(\d+\.?\d*)", "hemoglobin"),
    (r"\bPCV\b[\s\n\-:]*(\d+\.?\d*)", "pcv"),
    (r"\bMCV\b[\s\n\-:]*(\d+\.?\d*)", "mcv"),
    (r"\bMCH\b[\s\n\-:]*(\d+\.?\d*)", "mch"),
    (r"\bMCHC\b[\s\n\-:]*(\d+\.?\d*)", "mchc"),
    (r"\b(?:RDW|RDWCV|RDWSD)\b[\s\n\-:]*(\d+\.?\d*)", "rdw"),
    (r"\bNeutrophils?\b[\s\n\-:]*(\d+\.?\d*)", "neutrophils"),
    (r"\bLymphocytes?\b[\s\n\-:]*(\d+\.?\d*)", "lymphocytes"),
    (r"\bEosinophils?\b[\s\n\-:]*(\d+\.?\d*)", "eosinophils"),
    (r"\bMonocytes?\b[\s\n\-:]*(\d+\.?\d*)", "monocytes"),
    (r"\bBasophils?\b[\s\n\-:]*(\d+\.?\d*)", "basophils"),
    (r"\b(?:TSH|Thyroid\s+Stimulating\s+Hormone)(?:\s*\([^\)]*\))?[\s\n\-:=]*(\d+\.?\d*)", "tsh"),
    (r"\b(?:Free\s+T3|FT3|Free\s+Triiodothyronine)(?:\s*\([^\)]*\))?[\s\n\-:=]*(\d+\.?\d*)", "ft3"),
    (r"\b(?:Free\s+T4|FT4|Free\s+Thyroxine)(?:\s*\([^\)]*\))?[\s\n\-:=]*(\d+\.?\d*)", "ft4"),
    (r"(?<!free\s)(?<!ft)(?<!free-)(?<!\()\b(?:Total\s+Triiodothyronine|Total\s+T3|(?:(?<![a-zA-Z0-9])T3))(?:\s*\([^\)]*\))?[\s\n\-:=]*(\d+\.?\d*)", "t3"),
    (r"(?<!free\s)(?<!ft)(?<!free-)(?<!\()\b(?:Total\s+Thyroxine|Total\s+T4|(?:(?<![a-zA-Z0-9])T4))(?:\s*\([^\)]*\))?[\s\n\-:=]*(\d+\.?\d*)", "t4"),
    (r"creatinine(?:\s*\(serum\))?[\s\n\-:]*(\d+\.?\d*)", "creatinine"),
    (r"(?:total\s+)?cholesterol[\s\n\-:]*(\d+\.?\d*)", "cholesterol"),
    (r"LDL(?:\s*cholesterol)?[\s\n\-:]*(\d+\.?\d*)", "ldl"),
    (r"HDL(?:\s*cholesterol)?[\s\n\-:]*(\d+\.?\d*)", "hdl"),
    (r"triglycerides?[\s\n\-:]*(\d+\.?\d*)", "triglycerides"),
    (r"(?:Total\s+)?WBC(?:\s*Count)?[\s\n\-:]*(\d+[\.,]?\d*)", "wbc"),
    (r"RBC(?:\s*Count)?[\s\n\-:]*(\d+\.?\d*)", "rbc"),
    (r"Platelet(?:\s*Count)?s?[\s\n\-:]*(\d+[\.,]?\d*)", "platelets"),
    (r"SGPT(?:\s*\(alt\))?[\s\n\-:]*(\d+\.?\d*)", "sgpt"),
    (r"SGOT(?:\s*\(ast\))?[\s\n\-:]*(\d+\.?\d*)", "sgot"),
    (r"ALT[\s\n\-:]*(\d+\.?\d*)", "sgpt"),
    (r"AST[\s\n\-:]*(\d+\.?\d*)", "sgot"),
    (r"uric\s+acid[\s\n\-:]*(\d+\.?\d*)", "uric_acid"),
    (r"sodium[\s\n\-:]*(\d+\.?\d*)", "sodium"),
    (r"potassium[\s\n\-:]*(\d+\.?\d*)", "potassium"),
    (r"bilirubin(?:\s*\(total\))?[\s\n\-:]*(\d+\.?\d*)", "bilirubin"),
]


TEST_KEYWORDS = {
    "ft3": ["free t3", "ft3", "free triiodothyronine"],
    "ft4": ["free t4", "ft4", "free thyroxine"],
    "tsh": ["tsh", "thyroid stimulating hormone", "thyrotropin"],
    "t3": ["total t3", "triiodothyronine", "total triiodothyronine"],
    "t4": ["total t4", "total thyroxine"],
    "glucose": ["glucose", "blood sugar", "fasting sugar", "fbs"],
    "hba1c": ["hba1c", "glycated hemoglobin"],
    "hemoglobin": ["haemoglobin", "hemoglobin", "hb"],
    "pcv": ["pcv", "packed cell volume", "hematocrit"],
    "mcv": ["mcv"],
    "mch": ["mch"],
    "mchc": ["mchc"],
    "rdw": ["rdw", "rdwcv", "rdwsd"],
    "wbc": ["total wbc count", "wbc count", "wbc", "tlc", "white blood cell"],
    "neutrophils": ["neutrophils", "neutrophil"],
    "lymphocytes": ["lymphocytes", "lymphocyte"],
    "eosinophils": ["eosinophils", "eosinophil"],
    "monocytes": ["monocytes", "monocyte"],
    "basophils": ["basophils", "basophil"],
    "platelets": ["platelet count", "platelets", "platelet"],
    "rbc": ["rbc count", "rbc", "red blood cell"],
    "creatinine": ["creatinine", "serum creatinine"],
    "cholesterol": ["total cholesterol", "cholesterol"],
    "ldl": ["ldl cholesterol", "ldl"],
    "hdl": ["hdl cholesterol", "hdl"],
    "triglycerides": ["triglycerides", "triglyceride"],
    "sgpt": ["sgpt", "alt"],
    "sgot": ["sgot", "ast"],
    "bilirubin": ["total bilirubin", "bilirubin"],
    "uric_acid": ["uric acid"],
    "sodium": ["sodium"],
    "potassium": ["potassium"],
}


class ReportAnalyzer:
    def analyze(self, content: bytes, filename: str = "") -> Dict[str, Any]:
        text, block_values = self._extract_content_and_blocks(content, filename)
        if not text.strip() and not block_values:
            return {
                "error": "Could not extract text from file. Ensure it is a readable PDF or image.",
                "extracted_text": "",
                "lab_values": {},
                "abnormal_values": [],
                "normal_values": [],
                "summary": "Unable to process the report.",
                "recommendations": [],
            }

        # Combine block-level extraction with regex pattern extraction
        pattern_values = self._extract_lab_values(text)
        lab_values = {**pattern_values, **block_values}

        abnormal       = self._flag_abnormals(lab_values)
        abnormal_keys  = {a["test_key"] for a in abnormal}
        normal         = [k for k in lab_values if k not in abnormal_keys]
        summary        = self._generate_summary(lab_values, abnormal, filename)
        recommendations = self._generate_recommendations(abnormal)

        return {
            "extracted_text":  text[:2000],   # First 2000 chars for preview
            "lab_values":      lab_values,
            "abnormal_values": abnormal,
            "normal_values":   normal,
            "summary":         summary,
            "recommendations": recommendations,
            "report_type":     self._detect_report_type(text),
        }

    def _extract_content_and_blocks(self, content: bytes, filename: str):
        text = ""
        block_values = {}
        ext  = filename.lower().split(".")[-1] if "." in filename else "pdf"

        # Try PyMuPDF (for PDFs)
        if ext == "pdf" or not ext:
            try:
                import fitz  # PyMuPDF
                doc  = fitz.open(stream=content, filetype="pdf")
                pages_text = []
                for page in doc:
                    pages_text.append(page.get_text())
                    blocks = page.get_text("blocks")
                    for b in blocks:
                        b_text = b[4].strip()
                        lines = [l.strip() for l in b_text.split("\n") if l.strip()]
                        for test_key, aliases in TEST_KEYWORDS.items():
                            if test_key in block_values:
                                continue
                            for alias in aliases:
                                pattern = rf"\b{re.escape(alias)}\b"
                                if re.search(pattern, b_text, re.IGNORECASE):
                                    nums = []
                                    for line in lines:
                                        m = re.findall(r"(?<![A-Za-z0-9\.])(\d+\.?\d*)(?![A-Za-z0-9\.])", line)
                                        if m:
                                            nums.extend(m)
                                    if nums:
                                        try:
                                            block_values[test_key] = float(nums[0])
                                            break
                                        except Exception:
                                            pass
                doc.close()
                text = "\n".join(pages_text)
                if text.strip():
                    return text, block_values
            except Exception as e:
                print(f"PyMuPDF error: {e}")

        # OCR fallback for images or scanned PDFs
        try:
            from PIL import Image
            import pytesseract

            if ext in ("jpg", "jpeg", "png", "bmp", "tiff", "tif"):
                img  = Image.open(io.BytesIO(content))
                text = pytesseract.image_to_string(img)
            elif ext == "pdf":
                import fitz
                doc = fitz.open(stream=content, filetype="pdf")
                pages_text = []
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    pages_text.append(pytesseract.image_to_string(img))
                text = "\n".join(pages_text)
                doc.close()
        except Exception as e:
            print(f"OCR error: {e}")

        return text, block_values

    def _extract_lab_values(self, text: str) -> Dict[str, float]:
        lab_values = {}
        for pattern, key in EXTRACT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1).replace(",", ""))
                    if key not in lab_values:
                        lab_values[key] = val
                except ValueError:
                    pass
        return lab_values

    def _flag_abnormals(self, lab_values: Dict[str, float]) -> List[Dict]:
        abnormals = []
        for key, value in lab_values.items():
            if key not in NORMAL_RANGES:
                continue
            low, high, unit, label = NORMAL_RANGES[key]
            if value < low:
                abnormals.append({
                    "test_key": key,
                    "test":    label,
                    "value":   value,
                    "unit":    unit,
                    "status":  "LOW",
                    "normal":  f"{low}–{high} {unit}",
                    "flag":    "⬇️ Below Normal",
                })
            elif value > high:
                abnormals.append({
                    "test_key": key,
                    "test":    label,
                    "value":   value,
                    "unit":    unit,
                    "status":  "HIGH",
                    "normal":  f"{low}–{high} {unit}",
                    "flag":    "⬆️ Above Normal",
                })
        return abnormals

    def _generate_summary(self, lab_values, abnormals, filename) -> str:
        rtype = self._detect_report_type_from_values(lab_values)
        total = len(lab_values)
        abn   = len(abnormals)
        if total == 0:
            return "No standard lab values were detected in this report."
        summary = f"Report Analysis ({rtype}): {total} lab values extracted, "
        if abn == 0:
            summary += "all values are within normal range. No immediate concerns detected."
        elif abn == 1:
            summary += f"1 abnormal value detected: {abnormals[0]['test']} is {abnormals[0]['status']}."
        else:
            flags = ", ".join(f"{a['test']} ({a['status']})" for a in abnormals[:3])
            summary += f"{abn} abnormal values detected: {flags}."
        return summary

    def _generate_recommendations(self, abnormals: List[Dict]) -> List[str]:
        recs  = []
        tests = {a["test"] for a in abnormals}
        if "HbA1c" in tests or "Fasting Blood Glucose" in tests:
            recs.append("Elevated blood glucose/HbA1c — consult endocrinologist for diabetes management.")
        if "Total Cholesterol" in tests or "LDL Cholesterol" in tests:
            recs.append("Elevated lipid levels — dietary modification and statin therapy evaluation recommended.")
        if any("Hemoglobin" in t for t in tests):
            recs.append("Abnormal hemoglobin — further CBC and iron studies advised.")
        if "MCH" in tests or "MCHC" in tests or "MCV" in tests or "RDW" in tests:
            recs.append("Abnormal RBC indices (MCH/MCHC/MCV/RDW) — peripheral blood smear correlation and nutritional deficiency screen (B12/Folate/Iron) recommended.")
        if "Eosinophils" in tests:
            recs.append("Elevated Eosinophils — check for underlying allergic conditions, parasitic infections, or drug hypersensitivity.")
        if any(t in tests for t in ["TSH", "Total T3", "Total T4", "Free T3 (FT3)", "Free T4 (FT4)"]):
            recs.append("Abnormal thyroid panel (TSH / T3 / T4) — endocrinology consultation recommended to evaluate for hypothyroidism or hyperthyroidism (consider anti-TPO antibody screen).")
        if "Creatinine" in tests or "Creatinine (Male)" in tests or "Creatinine (Female)" in tests:
            recs.append("Elevated creatinine — renal function assessment and nephrology referral suggested.")
        if any("SGPT" in t or "SGOT" in t for t in tests):
            recs.append("Elevated liver enzymes — avoid hepatotoxic drugs; liver function monitoring advised.")
        if not recs:
            recs.append("All measured parameters are within normal range. Continue routine health monitoring.")
        recs.append("⚠️ This is an AI-assisted analysis. Always consult a qualified physician for diagnosis.")
        return recs

    def _detect_report_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(k in text_lower for k in ["hba1c", "glucose", "insulin", "glycated"]):
            return "Diabetes / Blood Sugar Report"
        if any(k in text_lower for k in ["hemoglobin", "hematocrit", "wbc", "rbc", "platelet", "cbc"]):
            return "Complete Blood Count (CBC)"
        if any(k in text_lower for k in ["tsh", "t3", "t4", "thyroid"]):
            return "Thyroid Function Test"
        if any(k in text_lower for k in ["creatinine", "urea", "bun", "gfr", "kidney"]):
            return "Kidney Function Test (KFT)"
        if any(k in text_lower for k in ["sgpt", "sgot", "alt", "ast", "bilirubin", "liver"]):
            return "Liver Function Test (LFT)"
        if any(k in text_lower for k in ["cholesterol", "ldl", "hdl", "triglycerides", "lipid"]):
            return "Lipid Profile"
        return "General Medical Report"

    def _detect_report_type_from_values(self, lab_values) -> str:
        if "hba1c" in lab_values or "glucose" in lab_values:
            return "Diabetes / Blood Sugar Report"
        if "hemoglobin" in lab_values or "wbc" in lab_values:
            return "Complete Blood Count (CBC)"
        if "tsh" in lab_values:
            return "Thyroid Function Test"
        if "creatinine" in lab_values:
            return "Kidney Function Test"
        if "sgpt" in lab_values or "sgot" in lab_values:
            return "Liver Function Test"
        if "cholesterol" in lab_values:
            return "Lipid Profile"
        return "General Medical Report"
