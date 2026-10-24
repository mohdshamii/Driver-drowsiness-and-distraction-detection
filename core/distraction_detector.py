"""
Distraction Detector Module using Face Position Centroid Tracking & Drift Approximation.
Estimates driver gaze diversion and looking away/down through bounding box movement
and spatial offset from the calibrated center of the video frame.
Clearly labeled as a Heuristic Approximation. No MediaPipe or neural networks.
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any
import config

class DistractionDetector:
    def __init__(self,
                 drift_x_thresh: float = config.HEAD_DRIFT_X_THRESHOLD,
                 drift_down_thresh: float = config.HEAD_DRIFT_DOWN_THRESHOLD,
                 consec_frames: int = config.DISTRACTION_CONSEC_FRAMES):
        self.drift_x_thresh = drift_x_thresh
        self.drift_down_thresh = drift_down_thresh
        self.consec_frames = consec_frames

        # State counters
        self.distraction_counter = 0
        self.total_distraction_events = 0
        self.no_face_counter = 0

        # Position tracking
        self.prev_centroid = None
        self.movement_velocity = 0.0
        self.smooth_norm_x = 0.0
        self.smooth_norm_y = 0.0
        self.alpha = 0.35  # Smoothing filter

    def update(self, face_data: Optional[Dict[str, Any]], frame_dims: Tuple[int, int], smooth: bool = True) -> Dict[str, Any]:
        """
        Calculates normalized centroid offset and movement drift.
        Determines distraction state with explicit Heuristic Approximation labeling.
        """
        w, h = frame_dims

        if face_data is None:
            self.no_face_counter += 1
            is_distracted = self.no_face_counter >= self.consec_frames
            if self.no_face_counter == self.consec_frames:
                self.total_distraction_events += 1

            return {
                "norm_drift_x": 0.0,
                "norm_drift_y": 0.0,
                "movement_speed": 0.0,
                "is_distracted": is_distracted,
                "distraction_type": "NO_FACE_IN_VIEW" if is_distracted else "NONE",
                "distraction_frames": self.no_face_counter,
                "total_distractions": self.total_distraction_events,
                "label": "DISTRACTION (Heuristic Approx: Head Drift/Turn)"
            }

        self.no_face_counter = 0
        cx, cy = face_data["centroid"]
        norm_x, norm_y = face_data["norm_pos"]

        # Smooth position (initialize directly on first frame or if smoothing disabled)
        if self.prev_centroid is None or not smooth:
            self.smooth_norm_x = norm_x
            self.smooth_norm_y = norm_y
        else:
            self.smooth_norm_x = self.alpha * norm_x + (1 - self.alpha) * self.smooth_norm_x
            self.smooth_norm_y = self.alpha * norm_y + (1 - self.alpha) * self.smooth_norm_y

        # Velocity tracking (pixels per frame)
        if self.prev_centroid is not None:
            dx = cx - self.prev_centroid[0]
            dy = cy - self.prev_centroid[1]
            self.movement_velocity = float(np.sqrt(dx**2 + dy**2))
        self.prev_centroid = (cx, cy)

        # Distraction classification criteria
        is_drift_x = abs(self.smooth_norm_x) > self.drift_x_thresh
        is_drift_down = self.smooth_norm_y > self.drift_down_thresh

        distraction_type = "NONE"
        if is_drift_down:
            distraction_type = "LOOKING_DOWN_PHONE"
        elif is_drift_x:
            distraction_type = "LOOKING_AWAY_RIGHT" if self.smooth_norm_x > 0 else "LOOKING_AWAY_LEFT"

        is_drifting = is_drift_x or is_drift_down

        if is_drifting:
            self.distraction_counter += 1
            if self.distraction_counter == self.consec_frames:
                self.total_distraction_events += 1
        else:
            self.distraction_counter = max(0, self.distraction_counter - 2)

        is_distracted = self.distraction_counter >= self.consec_frames

        return {
            "norm_drift_x": float(np.round(self.smooth_norm_x, 3)),
            "norm_drift_y": float(np.round(self.smooth_norm_y, 3)),
            "movement_speed": float(np.round(self.movement_velocity, 1)),
            "is_distracted": is_distracted,
            "distraction_type": distraction_type if is_distracted else "NONE",
            "distraction_frames": self.distraction_counter,
            "total_distractions": self.total_distraction_events,
            "label": "DISTRACTION (Heuristic Approx: Head Drift/Turn)"
        }

    def draw_annotations(self, canvas: np.ndarray,
                         face_data: Optional[Dict[str, Any]],
                         distract_data: Dict[str, Any]) -> np.ndarray:
        """
        Draws centroid drift direction vector and central attention reference box.
        Clearly labels approximation on live video feed.
        """
        h, w, _ = canvas.shape
        center_x, center_y = w // 2, h // 2

        # 1. Central Attention Reference Safe Zone (subtle box)
        safe_w = int(w * self.drift_x_thresh)
        safe_h = int(h * self.drift_down_thresh)
        cv2.rectangle(canvas,
                      (center_x - safe_w, center_y - safe_h),
                      (center_x + safe_w, center_y + safe_h),
                      (80, 90, 110), 1)

        if face_data is not None:
            cx, cy = face_data["centroid"]
            # Draw vector line from calibrated center to current driver centroid
            vec_color = (0, 0, 255) if distract_data.get("is_distracted") else (0, 240, 255)
            cv2.arrowedLine(canvas, (center_x, center_y), (cx, cy), vec_color, 2, tipLength=0.2)

        # 2. Disclaimer badge at bottom of video
        cv2.putText(canvas, "* DISTRACTION: Heuristic Head-Drift Approx", (15, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 160, 180), 1)

        return canvas

    def reset_stats(self):
        """Reset distraction statistics."""
        self.distraction_counter = 0
        self.total_distraction_events = 0
        self.no_face_counter = 0
        self.prev_centroid = None
        self.movement_velocity = 0.0
        self.smooth_norm_x = 0.0
        self.smooth_norm_y = 0.0
