"""
Machine Learning Classifier for Driver State Prediction.
Wraps the trained scikit-learn pipeline for real-time inference
on extracted landmark features.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import joblib
import config

class DriverStateClassifier:
    CLASS_NAMES = {
        0: "ALERT",
        1: "DROWSY",
        2: "YAWNING",
        3: "DISTRACTED"
    }

    FEATURE_NAMES = [
        "ear",
        "mar",
        "perclos",
        "blink_rate",
        "yaw",
        "pitch",
        "roll",
        "ear_mar_ratio"
    ]

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or config.MODEL_PATH
        self.model = None
        self.is_loaded = False
        self.load_model()

    def load_model(self) -> bool:
        """Loads serialized model pipeline if exists."""
        if self.model_path and Path(self.model_path).exists():
            try:
                self.model = joblib.load(str(self.model_path))
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"[ML Model] Error loading model from {self.model_path}: {e}")
                self.is_loaded = False
        return False

    def extract_feature_vector(self,
                               ear: float,
                               mar: float,
                               perclos: float,
                               blink_rate: float,
                               yaw: float,
                               pitch: float,
                               roll: float) -> np.ndarray:
        """Constructs standardized feature array."""
        ear_mar_ratio = float(ear / (mar + 1e-5))
        features = np.array([
            ear,
            mar,
            perclos,
            blink_rate,
            yaw,
            pitch,
            roll,
            ear_mar_ratio
        ], dtype=np.float32).reshape(1, -1)
        return features

    def predict(self,
                ear: float,
                mar: float,
                perclos: float,
                blink_rate: float,
                yaw: float,
                pitch: float,
                roll: float) -> Dict[str, Any]:
        """
        Runs ML inference on driver telemetry features.
        Returns predicted class, probabilities, and confidence score.
        """
        if not self.is_loaded or self.model is None:
            return {"status": "NOT_LOADED"}

        try:
            X = self.extract_feature_vector(ear, mar, perclos, blink_rate, yaw, pitch, roll)
            pred_id = int(self.model.predict(X)[0])
            pred_name = self.CLASS_NAMES.get(pred_id, "UNKNOWN")

            # Check if model supports predict_proba
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[0]
                prob_dict = {self.CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
                confidence = float(np.max(probs))
            else:
                prob_dict = {pred_name: 1.0}
                confidence = 1.0

            return {
                "status": "OK",
                "predicted_class_id": pred_id,
                "predicted_class_name": pred_name,
                "confidence": confidence,
                "probabilities": prob_dict
            }
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
