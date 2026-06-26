"""
SHAP Explainability Module
Uses KernelExplainer on the PyTorch disease classifier
"""
import numpy as np
from typing import Dict, Any

class SHAPExplainer:
    def __init__(self, classifier):
        self.clf = classifier
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is not None:
            return self._explainer
        try:
            import shap
            # Background dataset: all-zeros (no symptoms present)
            background = np.zeros((1, len(self.clf.feature_cols)), dtype=np.float32)
            self._explainer = shap.KernelExplainer(
                self.clf.predict_vector, background
            )
            return self._explainer
        except Exception as e:
            print(f"SHAP init warning: {e}")
            return None

    def explain(self, symptoms: list) -> Dict[str, Any]:
        """
        Returns top contributing features for the top predicted disease.
        """
        feature_cols = self.clf.feature_cols
        if not feature_cols:
            return {"features": [], "values": [], "disease": "Unknown"}

        vec = self.clf.symptoms_to_vector_np(symptoms).reshape(1, -1)
        top_pred = self.clf.predict(symptoms, top_k=1)
        top_disease = top_pred[0]["disease"] if top_pred else "Unknown"

        try:
            explainer = self._get_explainer()
            if explainer is None:
                raise ValueError("Explainer unavailable")

            shap_vals = explainer.shap_values(vec, nsamples=100)
            # shap_vals shape: (num_classes, 1, num_features) or (1, num_features)
            if isinstance(shap_vals, list):
                # Multi-class: find index of top disease
                disease_idx = (
                    self.clf.label_classes.index(top_disease)
                    if top_disease in self.clf.label_classes else 0
                )
                vals = np.array(shap_vals[disease_idx]).flatten()
            else:
                vals = np.array(shap_vals).flatten()

            # Top 10 absolute features
            top_indices = np.argsort(np.abs(vals))[::-1][:10]
            features = [feature_cols[i].replace("_", " ").title() for i in top_indices]
            values   = [round(float(vals[i]), 4) for i in top_indices]

        except Exception as e:
            print(f"SHAP compute warning: {e}")
            # Fallback: return present symptoms as top features
            present = [s.replace("_", " ").title() for s in symptoms[:10]]
            features = present
            values   = [1.0] * len(present)

        return {
            "disease":  top_disease,
            "features": features,
            "values":   values,
        }
