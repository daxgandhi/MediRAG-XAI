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
    "tsh":            (0.4,  4.0,  "mIU/L", "TSH"),
    "creatinine_m":   (0.74, 1.35, "mg/dL", "Creatinine (Male)"),
    "creatinine_f":   (0.59, 1.04, "mg/dL", "Creatinine (Female)"),
    "creatinine":     (0.59, 1.35, "mg/dL", "Creatinine"),
    "cholesterol":    (0,    200,  "mg/dL", "Total Cholesterol"),
    "ldl":            (0,    100,  "mg/dL", "LDL Cholesterol"),
    "hdl":            (40,   60,   "mg/dL", "HDL Cholesterol"),
    "triglycerides":  (0,    150,  "mg/dL", "Triglycerides"),
    "wbc":            (4500, 11000,"cells/µL","WBC"),
    "rbc":            (4.5,  5.5,  "M/µL",  "RBC"),
    "platelets":      (150000, 400000, "/µL","Platelets"),
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
    (r"hemoglobin(?:\s*\(hb\))?[\s\n\-:]*(\d+\.?\d*)", "hemoglobin"),
    (r"TSH[\s\n\-:]*(\d+\.?\d*)", "tsh"),
    (r"creatinine(?:\s*\(serum\))?[\s\n\-:]*(\d+\.?\d*)", "creatinine"),
    (r"(?:total\s+)?cholesterol[\s\n\-:]*(\d+\.?\d*)", "cholesterol"),
    (r"LDL(?:\s*cholesterol)?[\s\n\-:]*(\d+\.?\d*)", "ldl"),
    (r"HDL(?:\s*cholesterol)?[\s\n\-:]*(\d+\.?\d*)", "hdl"),
    (r"triglycerides?[\s\n\-:]*(\d+\.?\d*)", "triglycerides"),
    (r"WBC(?:\s*count)?[\s\n\-:]*(\d+[\.,]?\d*)", "wbc"),
    (r"RBC(?:\s*count)?[\s\n\-:]*(\d+\.?\d*)", "rbc"),
    (r"platelet(?:\s*count)?s?[\s\n\-:]*(\d+[\.,]?\d*)", "platelets"),
    (r"SGPT(?:\s*\(alt\))?[\s\n\-:]*(\d+\.?\d*)", "sgpt"),
    (r"SGOT(?:\s*\(ast\))?[\s\n\-:]*(\d+\.?\d*)", "sgot"),
    (r"ALT[\s\n\-:]*(\d+\.?\d*)", "sgpt"),
    (r"AST[\s\n\-:]*(\d+\.?\d*)", "sgot"),
    (r"uric\s+acid[\s\n\-:]*(\d+\.?\d*)", "uric_acid"),
    (r"sodium[\s\n\-:]*(\d+\.?\d*)", "sodium"),
    (r"potassium[\s\n\-:]*(\d+\.?\d*)", "potassium"),
    (r"bilirubin(?:\s*\(total\))?[\s\n\-:]*(\d+\.?\d*)", "bilirubin"),
]


class ReportAnalyzer:
    def analyze(self, content: bytes, filename: str = "") -> Dict[str, Any]:
        text = self._extract_text(content, filename)
        if not text.strip():
            return {
                "error": "Could not extract text from file. Ensure it is a readable PDF or image.",
                "extracted_text": "",
                "lab_values": {},
                "abnormal_values": [],
                "normal_values": [],
                "summary": "Unable to process the report.",
                "recommendations": [],
            }

        lab_values     = self._extract_lab_values(text)
        abnormal       = self._flag_abnormals(lab_values)
        normal         = [k for k in lab_values if k not in [a["test"] for a in abnormal]]
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

    def _extract_text(self, content: bytes, filename: str) -> str:
        text = ""
        ext  = filename.lower().split(".")[-1] if "." in filename else "pdf"

        # Try PyMuPDF (for PDFs)
        if ext == "pdf" or not ext:
            try:
                import fitz  # PyMuPDF
                doc  = fitz.open(stream=content, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                if text.strip():
                    return text
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
                # Convert PDF pages to images and OCR
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

        return text

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
                    "test":    label,
                    "value":   value,
                    "unit":    unit,
                    "status":  "LOW",
                    "normal":  f"{low}–{high} {unit}",
                    "flag":    "⬇️ Below Normal",
                })
            elif value > high:
                abnormals.append({
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
        if "TSH" in tests:
            recs.append("Abnormal TSH — thyroid function panel (T3/T4) recommended.")
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
