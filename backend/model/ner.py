"""
Clinical NER Module
Uses spaCy with SciSpacy models (or fallback to en_core_web_sm)
Extracts: DISEASE, DRUG, SYMPTOM, MEDICAL_HISTORY, LAB_VALUES
"""
import re
from typing import Dict, List

# Entity category rules (regex + keyword lists as fallback)
SYMPTOM_KEYWORDS = {
    "fever", "cough", "fatigue", "headache", "nausea", "vomiting",
    "dizziness", "chest pain", "shortness of breath", "rash", "itching",
    "swelling", "pain", "weakness", "chills", "night sweats", "weight loss",
    "weight gain", "appetite loss", "jaundice", "bleeding", "palpitations",
    "insomnia", "anxiety", "depression", "burning", "tingling", "numbness",
    "wheezing", "sneezing", "runny nose", "sore throat", "joint pain",
    "muscle pain", "back pain", "abdominal pain", "bloating", "diarrhea",
    "constipation", "frequent urination", "dark urine", "pale stools",
    "confusion", "memory loss", "tremors", "seizures", "blurred vision",
}

DRUG_KEYWORDS = {
    "metformin", "insulin", "aspirin", "ibuprofen", "paracetamol",
    "amoxicillin", "ciprofloxacin", "lisinopril", "atorvastatin",
    "omeprazole", "salbutamol", "prednisolone", "warfarin", "heparin",
    "levothyroxine", "amlodipine", "losartan", "metoprolol", "furosemide",
    "azithromycin", "doxycycline", "chloroquine", "artemisinin",
    "ceftriaxone", "vancomycin", "fluoxetine", "sertraline", "alprazolam",
    "diazepam", "morphine", "codeine", "tramadol", "gabapentin",
    "hydroxychloroquine", "remdesivir", "dexamethasone",
}

DISEASE_KEYWORDS = {
    "diabetes", "hypertension", "asthma", "copd", "pneumonia",
    "tuberculosis", "tb", "migraine", "anemia", "dengue", "malaria",
    "typhoid", "covid-19", "covid", "heart disease", "kidney disease",
    "liver disease", "thyroid", "arthritis", "gastritis", "gerd",
    "depression", "anxiety", "bronchitis", "sinusitis", "uti",
    "obesity", "psoriasis", "eczema", "hepatitis", "jaundice",
    "cancer", "stroke", "epilepsy", "parkinson", "alzheimer",
    "hypothyroidism", "hyperthyroidism", "hypoglycemia", "hypolipidemia",
    "cholesterol", "osteoporosis", "fibromyalgia", "lupus",
}

LAB_PATTERNS = [
    (r"HbA1c\s*[:\-]?\s*(\d+\.?\d*\s*%?)", "HbA1c"),
    (r"glucose\s*[:\-]?\s*(\d+\.?\d*\s*mg/dL?)", "Glucose"),
    (r"hemoglobin\s*[:\-]?\s*(\d+\.?\d*\s*g/dL?)", "Hemoglobin"),
    (r"TSH\s*[:\-]?\s*(\d+\.?\d*\s*\w+/?\w*)", "TSH"),
    (r"creatinine\s*[:\-]?\s*(\d+\.?\d*\s*mg/dL?)", "Creatinine"),
    (r"cholesterol\s*[:\-]?\s*(\d+\.?\d*\s*mg/dL?)", "Cholesterol"),
    (r"WBC\s*[:\-]?\s*(\d+\.?\d*\s*\w+)", "WBC"),
    (r"RBC\s*[:\-]?\s*(\d+\.?\d*\s*\w+)", "RBC"),
    (r"platelets?\s*[:\-]?\s*(\d+[\d,]*\s*\w+)", "Platelets"),
    (r"blood pressure\s*[:\-]?\s*(\d+/\d+\s*mmHg?)", "Blood Pressure"),
    (r"BP\s*[:\-]?\s*(\d+/\d+)", "Blood Pressure"),
    (r"heart rate\s*[:\-]?\s*(\d+\s*bpm?)", "Heart Rate"),
    (r"oxygen\s+saturation\s*[:\-]?\s*(\d+\s*%?)", "SpO2"),
    (r"SpO2\s*[:\-]?\s*(\d+\s*%?)", "SpO2"),
]


class ClinicalNER:
    def __init__(self):
        self.nlp = None
        self._load_model()

    def _load_model(self):
        """Try SciSpacy first, fall back to spaCy en_core_web_sm."""
        # Try en_core_sci_sm (SciSpacy)
        for model_name in ["en_core_sci_sm", "en_core_web_sm", "en_core_web_md"]:
            try:
                import spacy
                self.nlp = spacy.load(model_name)
                print(f"[OK] NER model loaded: {model_name}")
                return
            except OSError:
                continue
            except Exception as e:
                print(f"  Could not load {model_name}: {e}")
                continue
        print("[WARN] No spaCy model found. Using regex-only NER.")

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Extract clinical entities from free text."""
        entities = {
            "DISEASE":         [],
            "DRUG":            [],
            "SYMPTOM":         [],
            "MEDICAL_HISTORY": [],
            "LAB_VALUES":      [],
        }
        text_lower = text.lower()

        # 1. spaCy-based extraction
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    label = ent.label_.upper()
                    val   = ent.text.strip()
                    if label in ("DISEASE", "CONDITION"):
                        entities["DISEASE"].append(val)
                    elif label in ("DRUG", "CHEMICAL", "MEDICATION", "GPE"):
                        # GPE sometimes picks up drug names in SciSpacy
                        pass
                    elif label == "SYMPTOM":
                        entities["SYMPTOM"].append(val)
            except Exception:
                pass

        # 2. Keyword matching (always run as supplement)
        for kw in DISEASE_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower) and kw.title() not in entities["DISEASE"]:
                entities["DISEASE"].append(kw.title())
        for kw in DRUG_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower) and kw.title() not in entities["DRUG"]:
                entities["DRUG"].append(kw.title())
        for kw in SYMPTOM_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower) and kw.title() not in entities["SYMPTOM"]:
                entities["SYMPTOM"].append(kw.title())

        # 3. Regex-based lab value extraction
        for pattern, name in LAB_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities["LAB_VALUES"].append(f"{name}: {match.group(1).strip()}")

        # 4. Medical history patterns
        history_patterns = [
            r"history of\s+([\w\s]+?)(?:\.|,|and|$)",
            r"diagnosed with\s+([\w\s]+?)(?:\.|,|and|$)",
            r"known case of\s+([\w\s]+?)(?:\.|,|and|$)",
            r"suffering from\s+([\w\s]+?)(?:\.|,|and|$)",
        ]
        for pat in history_patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                condition = match.group(1).strip()
                if condition and condition not in entities["MEDICAL_HISTORY"]:
                    entities["MEDICAL_HISTORY"].append(condition.title())

        # Deduplicate
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        return entities
