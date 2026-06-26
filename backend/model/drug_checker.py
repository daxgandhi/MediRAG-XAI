"""
Drug Safety Knowledge Base
Checks contraindications, pregnancy warnings, disease interactions, allergy alerts
Based on established pharmacological references
"""
from typing import Dict, List, Any

# ─── Comprehensive Drug Knowledge Base ────────────────────────────────────────
DRUG_DB = {
    "metformin": {
        "display_name": "Metformin",
        "class": "Biguanide / Antidiabetic",
        "indications": ["Type 2 Diabetes", "Prediabetes", "PCOS"],
        "contraindications": ["Renal Failure (eGFR <30)", "Liver Failure", "Alcoholism", "Lactic Acidosis History"],
        "pregnancy_category": "B",
        "pregnancy_warning": "Generally considered safe in pregnancy (especially for GDM), but use with caution in third trimester. Consult physician.",
        "disease_interactions": {
            "Kidney Disease": "CONTRAINDICATED — risk of lactic acidosis",
            "Liver Disease": "AVOID — impaired lactate clearance",
            "Heart Failure": "Use with caution — reduced renal perfusion risk",
        },
        "drug_interactions": ["Alcohol", "Iodine Contrast Dye", "Cimetidine", "Topiramate"],
        "allergy_warnings": [],
        "side_effects": ["Nausea", "Diarrhea", "Lactic Acidosis (rare)", "Vitamin B12 deficiency"],
        "monitoring": ["Renal Function (eGFR)", "Vitamin B12 levels", "Blood Glucose"],
    },
    "lisinopril": {
        "display_name": "Lisinopril",
        "class": "ACE Inhibitor / Antihypertensive",
        "indications": ["Hypertension", "Heart Failure", "Diabetic Nephropathy"],
        "contraindications": ["Pregnancy", "Bilateral Renal Artery Stenosis", "Hyperkalemia", "Angioedema History"],
        "pregnancy_category": "D",
        "pregnancy_warning": "⚠️ CONTRAINDICATED IN PREGNANCY — can cause fetal renal dysgenesis, oligohydramnios, neonatal hypotension, and death.",
        "disease_interactions": {
            "Kidney Disease": "Use with caution — monitor serum potassium and creatinine",
            "Hyperkalemia": "CONTRAINDICATED",
            "Renal Artery Stenosis": "CONTRAINDICATED",
        },
        "drug_interactions": ["Potassium-sparing Diuretics", "NSAIDs", "Lithium", "Aliskiren"],
        "allergy_warnings": ["ACE Inhibitor hypersensitivity", "Angioedema to any ACE inhibitor"],
        "side_effects": ["Dry Cough", "Hyperkalemia", "Hypotension", "Angioedema"],
        "monitoring": ["Blood Pressure", "Serum Potassium", "Renal Function"],
    },
    "aspirin": {
        "display_name": "Aspirin (Acetylsalicylic Acid)",
        "class": "NSAID / Antiplatelet",
        "indications": ["Pain Relief", "Fever", "Anti-inflammatory", "Antiplatelet (cardiovascular)"],
        "contraindications": ["Active Peptic Ulcer", "Bleeding Disorders", "Reye Syndrome risk (children)", "Aspirin-sensitive Asthma"],
        "pregnancy_category": "C/D",
        "pregnancy_warning": "Avoid in third trimester — may cause premature closure of ductus arteriosus and maternal/neonatal bleeding.",
        "disease_interactions": {
            "Peptic Ulcer Disease": "CONTRAINDICATED — risk of GI bleeding",
            "Asthma": "WARNING — may precipitate bronchospasm in sensitive patients",
            "Kidney Disease": "Avoid chronic use — can worsen renal function",
            "Liver Disease": "Use with extreme caution",
        },
        "drug_interactions": ["Warfarin", "Clopidogrel", "Ibuprofen", "Methotrexate", "SSRIs"],
        "allergy_warnings": ["Aspirin allergy", "NSAID hypersensitivity", "Sulfite allergy"],
        "side_effects": ["GI Upset", "Tinnitus", "Bleeding", "Reye Syndrome (children)"],
        "monitoring": ["Signs of bleeding", "Renal function with chronic use"],
    },
    "ibuprofen": {
        "display_name": "Ibuprofen",
        "class": "NSAID / Anti-inflammatory",
        "indications": ["Pain", "Fever", "Inflammation", "Arthritis"],
        "contraindications": ["Peptic Ulcer", "Renal Impairment", "Heart Failure", "Third Trimester Pregnancy"],
        "pregnancy_category": "C/D",
        "pregnancy_warning": "AVOID in third trimester — risk of premature closure of ductus arteriosus. Use with caution in first/second trimester.",
        "disease_interactions": {
            "Kidney Disease": "AVOID — NSAIDs reduce renal blood flow",
            "Heart Disease": "Increases cardiovascular risk with chronic use",
            "GERD / Peptic Ulcer": "CONTRAINDICATED without gastroprotection",
            "Hypertension": "May increase blood pressure",
        },
        "drug_interactions": ["Warfarin", "Aspirin", "Lithium", "ACE Inhibitors", "Methotrexate"],
        "allergy_warnings": ["NSAID hypersensitivity", "Aspirin allergy"],
        "side_effects": ["GI Distress", "Ulcers", "Renal Toxicity", "Cardiovascular Risk"],
        "monitoring": ["Renal function", "GI symptoms"],
    },
    "amoxicillin": {
        "display_name": "Amoxicillin",
        "class": "Penicillin Antibiotic",
        "indications": ["Bacterial Infections", "Pneumonia", "UTI", "Sinusitis", "Strep Throat"],
        "contraindications": ["Penicillin Allergy", "Mononucleosis (rash risk)"],
        "pregnancy_category": "B",
        "pregnancy_warning": "Generally safe in pregnancy. Preferred antibiotic for many infections during pregnancy.",
        "disease_interactions": {
            "Kidney Disease": "Dose adjustment required",
            "Liver Disease": "Use with caution",
        },
        "drug_interactions": ["Warfarin", "Oral Contraceptives", "Allopurinol", "Methotrexate"],
        "allergy_warnings": ["Penicillin allergy", "Cephalosporin allergy (cross-reactivity ~1-2%)", "Mold allergy"],
        "side_effects": ["Diarrhea", "Nausea", "Rash", "Anaphylaxis (rare)"],
        "monitoring": ["Signs of allergic reaction", "Renal function"],
    },
    "atorvastatin": {
        "display_name": "Atorvastatin",
        "class": "Statin / HMG-CoA Reductase Inhibitor",
        "indications": ["Hypercholesterolemia", "Cardiovascular Disease Prevention"],
        "contraindications": ["Pregnancy", "Active Liver Disease", "Unexplained elevated transaminases"],
        "pregnancy_category": "X",
        "pregnancy_warning": "⚠️ ABSOLUTELY CONTRAINDICATED IN PREGNANCY — causes fetal harm, must discontinue before conception.",
        "disease_interactions": {
            "Liver Disease": "CONTRAINDICATED — hepatotoxicity risk",
            "Kidney Disease": "Use with caution; myopathy risk increased",
        },
        "drug_interactions": ["Gemfibrozil", "Cyclosporine", "Clarithromycin", "Niacin", "Digoxin"],
        "allergy_warnings": [],
        "side_effects": ["Myopathy", "Rhabdomyolysis", "Hepatotoxicity", "Muscle Pain", "Elevated CK"],
        "monitoring": ["Liver function tests", "CK levels", "Lipid panel"],
    },
    "warfarin": {
        "display_name": "Warfarin",
        "class": "Anticoagulant / Vitamin K Antagonist",
        "indications": ["DVT", "Atrial Fibrillation", "Pulmonary Embolism", "Mechanical Heart Valves"],
        "contraindications": ["Pregnancy", "Active Bleeding", "Recent Surgery", "Severe Hypertension"],
        "pregnancy_category": "X",
        "pregnancy_warning": "⚠️ CONTRAINDICATED IN PREGNANCY — causes warfarin embryopathy, fetal bleeding, and can be teratogenic.",
        "disease_interactions": {
            "Liver Disease": "Increased bleeding risk — avoid or use with extreme caution",
            "Kidney Disease": "Increased bleeding risk",
            "Hypertension": "Uncontrolled hypertension increases stroke risk",
        },
        "drug_interactions": ["Aspirin", "NSAIDs", "Antibiotics", "Antifungals", "SSRIs", "Amiodarone", "Clopidogrel"],
        "allergy_warnings": [],
        "side_effects": ["Bleeding", "Skin Necrosis", "Purple Toe Syndrome"],
        "monitoring": ["INR (target 2-3)", "Signs of bleeding", "Drug and food interactions"],
    },
    "omeprazole": {
        "display_name": "Omeprazole",
        "class": "Proton Pump Inhibitor (PPI)",
        "indications": ["GERD", "Peptic Ulcer", "H. pylori Eradication", "Zollinger-Ellison Syndrome"],
        "contraindications": ["Hypersensitivity to PPIs"],
        "pregnancy_category": "C",
        "pregnancy_warning": "Use only if clearly needed during pregnancy. Limited data available.",
        "disease_interactions": {
            "Liver Disease": "Dose reduction may be required",
        },
        "drug_interactions": ["Clopidogrel (reduced efficacy)", "Methotrexate", "Rilpivirine", "Atazanavir"],
        "allergy_warnings": ["PPI hypersensitivity"],
        "side_effects": ["Headache", "Nausea", "Hypomagnesemia (long-term)", "C. difficile risk", "Bone fracture risk"],
        "monitoring": ["Magnesium levels (long-term)", "Signs of C. diff", "Bone density"],
    },
    "insulin": {
        "display_name": "Insulin",
        "class": "Antidiabetic Hormone",
        "indications": ["Type 1 Diabetes", "Type 2 Diabetes", "Diabetic Ketoacidosis", "Gestational Diabetes"],
        "contraindications": ["Hypoglycemia"],
        "pregnancy_category": "B",
        "pregnancy_warning": "Preferred antidiabetic agent during pregnancy. Requires careful glucose monitoring.",
        "disease_interactions": {
            "Kidney Disease": "Dose reduction often needed — reduced insulin clearance",
            "Liver Disease": "Unpredictable effects on glucose metabolism",
            "Heart Failure": "May cause fluid retention",
        },
        "drug_interactions": ["Beta-blockers (mask hypoglycemia symptoms)", "Glucocorticoids", "Thiazides", "Alcohol"],
        "allergy_warnings": ["Insulin allergy (rare)", "Latex allergy (some pen needles)"],
        "side_effects": ["Hypoglycemia", "Weight Gain", "Injection Site Lipodystrophy", "Hypokalemia"],
        "monitoring": ["Blood glucose (fasting/postprandial)", "HbA1c", "Serum potassium"],
    },
    "prednisolone": {
        "display_name": "Prednisolone",
        "class": "Corticosteroid",
        "indications": ["Asthma", "Rheumatoid Arthritis", "Allergic Reactions", "Autoimmune Diseases", "Inflammation"],
        "contraindications": ["Systemic Fungal Infections", "Live Vaccines during therapy"],
        "pregnancy_category": "C",
        "pregnancy_warning": "Use lowest effective dose. May cause neonatal adrenal suppression; associated with cleft palate in first trimester.",
        "disease_interactions": {
            "Diabetes": "May worsen glycemic control significantly",
            "Hypertension": "May worsen blood pressure",
            "Tuberculosis": "CAUTION — can reactivate latent TB",
            "Osteoporosis": "Accelerates bone loss",
            "Peptic Ulcer": "Increased risk of GI bleeding",
        },
        "drug_interactions": ["NSAIDs", "Warfarin", "Antidiabetics", "Antifungals", "Vaccines"],
        "allergy_warnings": ["Corticosteroid hypersensitivity"],
        "side_effects": ["Hyperglycemia", "Hypertension", "Cushing Syndrome", "Osteoporosis", "Immunosuppression"],
        "monitoring": ["Blood glucose", "Blood pressure", "Bone density", "Eye pressure"],
    },
    "levothyroxine": {
        "display_name": "Levothyroxine (T4)",
        "class": "Thyroid Hormone Replacement",
        "indications": ["Hypothyroidism", "Thyroid Cancer", "Goiter"],
        "contraindications": ["Untreated Adrenal Insufficiency", "Uncorrected Thyrotoxicosis"],
        "pregnancy_category": "A",
        "pregnancy_warning": "Safe and often required during pregnancy. Dose usually needs to increase 25-50% during pregnancy. Monitor TSH closely.",
        "disease_interactions": {
            "Heart Disease": "Use with caution — can precipitate angina/arrhythmia",
            "Diabetes": "May alter insulin requirements",
            "Adrenal Insufficiency": "Treat adrenal insufficiency first",
        },
        "drug_interactions": ["Calcium supplements", "Iron supplements", "Cholestyramine", "Antacids", "Soy"],
        "allergy_warnings": [],
        "side_effects": ["Palpitations", "Weight Loss", "Insomnia", "Tremors", "Heat Intolerance (overdose)"],
        "monitoring": ["TSH levels", "Free T4", "Heart rate"],
    },
    "metoprolol": {
        "display_name": "Metoprolol",
        "class": "Beta-1 Selective Blocker",
        "indications": ["Hypertension", "Heart Failure", "Angina", "Arrhythmia", "MI Secondary Prevention"],
        "contraindications": ["Cardiogenic Shock", "Severe Bradycardia", "2nd/3rd Degree AV Block", "Decompensated Heart Failure", "Sick Sinus Syndrome"],
        "pregnancy_category": "C",
        "pregnancy_warning": "Use with caution. Can cause neonatal bradycardia, hypoglycemia, and respiratory depression.",
        "disease_interactions": {
            "Asthma/COPD": "WARNING — can cause bronchospasm; use with extreme caution",
            "Diabetes": "May mask hypoglycemia symptoms",
            "Bradycardia": "CONTRAINDICATED",
        },
        "drug_interactions": ["Verapamil", "Diltiazem", "Digoxin", "Clonidine", "Antidiabetics"],
        "allergy_warnings": [],
        "side_effects": ["Bradycardia", "Fatigue", "Dizziness", "Bronchospasm", "Depression"],
        "monitoring": ["Heart rate", "Blood pressure", "ECG"],
    },
    "ciprofloxacin": {
        "display_name": "Ciprofloxacin",
        "class": "Fluoroquinolone Antibiotic",
        "indications": ["UTI", "Respiratory Infections", "Anthrax", "Typhoid", "Gastroenteritis"],
        "contraindications": ["Pregnancy", "Children under 18 (except anthrax/complicated UTI)", "Tendon disorders", "Myasthenia Gravis"],
        "pregnancy_category": "C",
        "pregnancy_warning": "AVOID during pregnancy — associated with fetal harm and musculoskeletal effects.",
        "disease_interactions": {
            "Kidney Disease": "Dose adjustment required",
            "Seizure Disorders": "May lower seizure threshold",
            "Myasthenia Gravis": "CONTRAINDICATED",
        },
        "drug_interactions": ["Antacids", "Iron", "Calcium", "Warfarin", "Theophylline", "NSAIDs"],
        "allergy_warnings": ["Fluoroquinolone allergy", "Quinolone hypersensitivity"],
        "side_effects": ["Tendinopathy", "QT Prolongation", "CNS Effects", "GI Disturbances", "Photosensitivity"],
        "monitoring": ["Tendon pain", "QTc interval", "Renal function"],
    },
    "furosemide": {
        "display_name": "Furosemide",
        "class": "Loop Diuretic",
        "indications": ["Heart Failure", "Edema", "Hypertension", "Renal Failure"],
        "contraindications": ["Anuria", "Severe Hypokalemia", "Sulfonamide Allergy"],
        "pregnancy_category": "C",
        "pregnancy_warning": "Use only if clearly necessary during pregnancy. May cause fetal electrolyte imbalances.",
        "disease_interactions": {
            "Kidney Disease": "Use with caution — may worsen prerenal azotemia",
            "Diabetes": "May worsen glucose control",
            "Gout": "May precipitate gout attacks",
            "Liver Cirrhosis": "Risk of electrolyte disturbances",
        },
        "drug_interactions": ["Digoxin", "Lithium", "Aminoglycosides", "NSAIDs", "ACE Inhibitors"],
        "allergy_warnings": ["Sulfonamide allergy"],
        "side_effects": ["Hypokalemia", "Hyponatremia", "Dehydration", "Ototoxicity (high doses)", "Hyperuricemia"],
        "monitoring": ["Electrolytes", "Renal function", "Blood pressure"],
    },
    "hydroxychloroquine": {
        "display_name": "Hydroxychloroquine",
        "class": "Antimalarial / DMARD",
        "indications": ["Malaria Prevention", "Lupus", "Rheumatoid Arthritis"],
        "contraindications": ["Retinopathy", "G6PD Deficiency", "Porphyria"],
        "pregnancy_category": "C",
        "pregnancy_warning": "May be used in pregnancy for lupus/RA under physician supervision. Discontinuing may worsen disease.",
        "disease_interactions": {
            "Diabetes": "May enhance hypoglycemic effects of antidiabetics",
        },
        "drug_interactions": ["QT-prolonging drugs", "Antacids", "Ciclosporin", "Digoxin"],
        "allergy_warnings": ["Aminoquinoline allergy"],
        "side_effects": ["Retinopathy (long-term)", "GI Disturbances", "QT Prolongation", "Hypoglycemia"],
        "monitoring": ["Annual eye exam (retina)", "Blood glucose", "ECG"],
    },
}


class DrugChecker:
    def __init__(self):
        self.db = DRUG_DB

    def check(
        self,
        drug_name: str,
        conditions: List[str] = None,
        allergies: List[str] = None,
        is_pregnant: bool = False,
        other_drugs: List[str] = None,
    ) -> Dict[str, Any]:
        conditions  = conditions  or []
        allergies   = allergies   or []
        other_drugs = other_drugs or []

        # Normalize drug name
        key = drug_name.lower().strip()
        drug_info = self._find_drug(key)

        if not drug_info:
            return {
                "drug_found":    False,
                "drug_name":     drug_name,
                "alerts":        [],
                "info":          {},
                "message":       f"'{drug_name}' is not in the current knowledge base. Please consult a pharmacist or prescribing physician.",
                "available_drugs": sorted(self.db.keys()),
            }

        alerts = []

        # 1. Pregnancy check
        if is_pregnant:
            pw  = drug_info.get("pregnancy_warning", "")
            cat = drug_info.get("pregnancy_category", "?")
            if cat in ("X", "D"):
                alerts.append({
                    "type":     "CRITICAL",
                    "category": "Pregnancy Contraindication",
                    "message":  drug_info.get("pregnancy_warning", f"Category {cat}: Contraindicated in pregnancy."),
                    "severity": "CRITICAL",
                })
            elif cat in ("C",):
                alerts.append({
                    "type":     "WARNING",
                    "category": "Pregnancy Warning",
                    "message":  pw or f"Category {cat}: Use only if benefits outweigh risks in pregnancy.",
                    "severity": "MODERATE",
                })
            elif pw:
                alerts.append({
                    "type":     "INFO",
                    "category": "Pregnancy Note",
                    "message":  pw,
                    "severity": "LOW",
                })

        # 2. Disease interaction check
        di = drug_info.get("disease_interactions", {})
        for condition in conditions:
            for disease_key, interaction_msg in di.items():
                if condition.lower() in disease_key.lower() or disease_key.lower() in condition.lower():
                    severity = "CRITICAL" if "CONTRAINDICATED" in interaction_msg.upper() else "WARNING"
                    alerts.append({
                        "type":     severity,
                        "category": "Disease Interaction",
                        "message":  f"{drug_info['display_name']} + {condition}: {interaction_msg}",
                        "severity": severity,
                    })

        # 3. Allergy check
        for allergy in allergies:
            for allergy_warning in drug_info.get("allergy_warnings", []):
                if allergy.lower() in allergy_warning.lower() or allergy_warning.lower() in allergy.lower():
                    alerts.append({
                        "type":     "CRITICAL",
                        "category": "Allergy Alert",
                        "message":  f"Patient has {allergy} allergy. {allergy_warning} documented for {drug_info['display_name']}.",
                        "severity": "CRITICAL",
                    })

        # 4. Drug-drug interaction check
        drug_di = drug_info.get("drug_interactions", [])
        for other in other_drugs:
            for ddi in drug_di:
                if other.lower() in ddi.lower() or ddi.lower() in other.lower():
                    alerts.append({
                        "type":     "WARNING",
                        "category": "Drug Interaction",
                        "message":  f"{drug_info['display_name']} may interact with {other}. Monitor closely.",
                        "severity": "MODERATE",
                    })

        # 5. Contraindication check
        for condition in conditions:
            for ci in drug_info.get("contraindications", []):
                if condition.lower() in ci.lower() or ci.lower() in condition.lower():
                    alerts.append({
                        "type":     "CRITICAL",
                        "category": "Contraindication",
                        "message":  f"{drug_info['display_name']} is contraindicated in: {ci}",
                        "severity": "CRITICAL",
                    })

        # Deduplicate alerts
        seen, unique_alerts = set(), []
        for a in alerts:
            key_str = a["message"]
            if key_str not in seen:
                seen.add(key_str)
                unique_alerts.append(a)

        disclaimer = (
            "⚠️ IMPORTANT: This tool provides reference information only. "
            "Never prescribe or modify medications without a licensed physician. "
            "Always consult qualified healthcare professionals for clinical decisions."
        )

        return {
            "drug_found":    True,
            "drug_name":     drug_info["display_name"],
            "drug_class":    drug_info.get("class", ""),
            "indications":   drug_info.get("indications", []),
            "alerts":        unique_alerts,
            "alert_count":   len(unique_alerts),
            "side_effects":  drug_info.get("side_effects", []),
            "monitoring":    drug_info.get("monitoring", []),
            "pregnancy_category": drug_info.get("pregnancy_category", "?"),
            "disclaimer":    disclaimer,
        }

    def _find_drug(self, name: str):
        # Exact match
        if name in self.db:
            return self.db[name]
        # Partial match
        for key, val in self.db.items():
            if name in key or key in name:
                return val
            if name in val.get("display_name", "").lower():
                return val
        return None
