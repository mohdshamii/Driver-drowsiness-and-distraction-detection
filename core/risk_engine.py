"""
Multi-factor Heuristic Risk Engine for Driver State Assessment.
Fuses OpenCV Haar eye visibility, mouth opening metric, and head drift
into a normalized Driver Risk Score (0-100%).
No MediaPipe or neural networks.
"""
from typing import Dict, Any, Optional
import numpy as np
import config

class RiskEngine:
    def __init__(self,
                 weight_eyes: float = config.RISK_WEIGHT_EYES,
                 weight_drift: float = config.RISK_WEIGHT_DRIFT,
                 weight_yawn: float = config.RISK_WEIGHT_YAWN):
        self.w_eyes = weight_eyes
        self.w_drift = weight_drift
        self.w_yawn = weight_yawn

        self.current_risk_score = 0.0
        self.max_session_risk = 0.0

    def evaluate(self,
                 drowsy_data: Dict[str, Any],
                 distract_data: Dict[str, Any],
                 smooth: bool = True) -> Dict[str, Any]:
        """
        Calculates unified driver risk score and categorical state.
        """
        # 1. Eye Closure Component (0 - 100)
        eye_absence_frames = drowsy_data.get("eye_absence_frames", 0)
        vis_ratio = drowsy_data.get("visibility_ratio", 100.0)

        if eye_absence_frames == 0:
            eye_subscore = 0.0
        elif eye_absence_frames < config.EYE_ABSENCE_CONSEC_FRAMES:
            eye_subscore = (eye_absence_frames / config.EYE_ABSENCE_CONSEC_FRAMES) * 50.0
        else:
            # Prolonged eye absence escalates rapidly
            extra = eye_absence_frames - config.EYE_ABSENCE_CONSEC_FRAMES
            eye_subscore = min(100.0, 50.0 + (extra / config.EYE_ABSENCE_CONSEC_FRAMES) * 50.0)

        # Penalty if low visibility over sliding window
        if vis_ratio < 40.0:
            eye_subscore = max(eye_subscore, (1.0 - (vis_ratio / 40.0)) * 60.0)

        # 2. Yawning Component (0 - 100)
        mouth_metric = drowsy_data.get("mouth_opening_metric", 0.0)
        mouth_frames = drowsy_data.get("mouth_open_frames", 0)

        if mouth_frames == 0:
            yawn_subscore = 0.0
        elif mouth_frames < config.YAWN_CONSEC_FRAMES:
            yawn_subscore = (mouth_frames / config.YAWN_CONSEC_FRAMES) * 50.0
        else:
            yawn_subscore = min(90.0, 50.0 + ((mouth_frames - config.YAWN_CONSEC_FRAMES) / config.YAWN_CONSEC_FRAMES) * 40.0)

        # 3. Distraction / Head Drift Component (0 - 100)
        distract_frames = distract_data.get("distraction_frames", 0)
        dtype = distract_data.get("distraction_type", "NONE")

        if distract_frames == 0:
            drift_subscore = 0.0
        elif distract_frames < config.DISTRACTION_CONSEC_FRAMES:
            drift_subscore = (distract_frames / config.DISTRACTION_CONSEC_FRAMES) * 55.0
        else:
            drift_subscore = min(100.0, 55.0 + ((distract_frames - config.DISTRACTION_CONSEC_FRAMES) / config.DISTRACTION_CONSEC_FRAMES) * 45.0)

        # Extra penalty if looking down at smartphone
        if dtype == "LOOKING_DOWN_PHONE":
            drift_subscore = min(100.0, drift_subscore * 1.2)

        # Fused Raw Risk Score
        raw_risk = (
            self.w_eyes * eye_subscore +
            self.w_drift * drift_subscore +
            self.w_yawn * yawn_subscore
        )

        # Exponential smoothing (alpha = 0.25)
        if self.current_risk_score == 0.0 or not smooth:
            self.current_risk_score = raw_risk
        else:
            self.current_risk_score = 0.25 * raw_risk + 0.75 * self.current_risk_score

        rounded_risk = float(np.clip(np.round(self.current_risk_score, 1), 0.0, 100.0))
        self.max_session_risk = max(self.max_session_risk, rounded_risk)

        # Status categorization
        if rounded_risk < config.RISK_LEVEL_NORMAL:
            driver_status = "ALERT"
            status_level = "NORMAL"
        elif rounded_risk < config.RISK_LEVEL_WARNING:
            if drowsy_data.get("is_yawning"):
                driver_status = "YAWNING / FATIGUE"
            elif distract_data.get("is_distracted"):
                driver_status = "DISTRACTED (APPROX)"
            else:
                driver_status = "MILD DROWSINESS"
            status_level = "WARNING"
        else:
            if drowsy_data.get("is_drowsy") and distract_data.get("is_distracted"):
                driver_status = "CRITICAL: DROWSY + DISTRACTED"
            elif drowsy_data.get("is_drowsy"):
                driver_status = "DROWSY: EYES CLOSED"
            elif distract_data.get("is_distracted"):
                driver_status = f"DISTRACTED: {dtype} (APPROX)"
            else:
                driver_status = "HIGH RISK"
            status_level = "CRITICAL"

        return {
            "risk_score": rounded_risk,
            "max_risk_score": float(np.round(self.max_session_risk, 1)),
            "status": driver_status,
            "status_level": status_level,
            "subscores": {
                "eyes": float(np.round(eye_subscore, 1)),
                "drift": float(np.round(drift_subscore, 1)),
                "yawn": float(np.round(yawn_subscore, 1))
            }
        }

    def reset(self):
        """Reset risk score accumulator."""
        self.current_risk_score = 0.0
        self.max_session_risk = 0.0
