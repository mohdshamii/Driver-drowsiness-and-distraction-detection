"""
Drowsiness and Yawning Detector Module using Pure OpenCV.
1. Detects eyes using OpenCV Haar Cascade (haarcascade_eye.xml) and tracks eye visibility over time.
2. Estimates yawning using mouth-region image processing (contour & oral cavity aspect ratio analysis).
No MediaPipe or neural networks.
"""
import os
import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import config

class DrowsinessDetector:
    def __init__(self,
                 eye_cascade_path: Optional[str] = None,
                 eye_absence_consec: int = config.EYE_ABSENCE_CONSEC_FRAMES,
                 yawn_threshold: float = config.YAWN_OPENING_THRESHOLD,
                 yawn_consec_frames: int = config.YAWN_CONSEC_FRAMES):
        self.eye_cascade_path = str(eye_cascade_path or config.EYE_CASCADE_PATH)
        if not os.path.exists(self.eye_cascade_path):
            raise FileNotFoundError(f"Eye cascade XML file not found at: {self.eye_cascade_path}")

        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_path)
        if self.eye_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV eye cascade from: {self.eye_cascade_path}")

        self.eye_absence_consec = eye_absence_consec
        self.yawn_threshold = yawn_threshold
        self.yawn_consec_frames = yawn_consec_frames

        # Eye visibility & blink state tracking
        self.eye_absence_frames = 0
        self.total_blinks = 0
        self.blink_in_progress = False
        self.current_blink_duration = 0
        self.total_drowsy_events = 0

        # Eye visibility history buffer (1 = eyes detected, 0 = eyes closed/undetected)
        self.visibility_buffer = deque(maxlen=config.EYE_VISIBILITY_WINDOW)

        # Yawning state tracking
        self.mouth_open_frames = 0
        self.total_yawns = 0
        self.is_currently_yawning = False
        self.yawn_cooldown = 0

        # Cached metrics for display
        self.last_detected_eyes = []
        self.last_mouth_metric = 0.0
        self.last_mouth_contour = None

    def detect_eyes_in_roi(self, eye_gray: np.ndarray,
                           roi_offset: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Runs Haar cascade inside the upper-face eye ROI.
        Returns list of eye bounding boxes in global frame coordinates (x, y, w, h).
        """
        if eye_gray is None or eye_gray.size == 0:
            return []

        roi_x, roi_y = roi_offset
        h, w = eye_gray.shape

        # Equalize ROI contrast
        eq_eye = cv2.equalizeHist(eye_gray)

        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            eq_eye,
            scaleFactor=1.12,
            minNeighbors=4,
            minSize=(int(w * 0.12), int(h * 0.18)),
            maxSize=(int(w * 0.45), int(h * 0.75))
        )

        global_eyes = []
        for (ex, ey, ew, eh) in eyes:
            global_eyes.append((roi_x + ex, roi_y + ey, ew, eh))

        # Keep at most 2 most prominent eyes
        if len(global_eyes) > 2:
            global_eyes = sorted(global_eyes, key=lambda b: b[2] * b[3], reverse=True)[:2]

        return global_eyes

    def analyze_mouth_opening(self, mouth_gray: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
        """
        Estimates yawning through mouth-region thresholding and contour analysis.
        When mouth opens in a yawn, a dark oral cavity forms.
        Calculates Mouth Opening Metric (MOM) = cavity height / mouth ROI width.
        """
        if mouth_gray is None or mouth_gray.size == 0:
            return 0.0, None

        h, w = mouth_gray.shape

        # 1. Blur to eliminate lip texture noise
        blurred = cv2.GaussianBlur(mouth_gray, (7, 7), 0)

        # 2. Adaptive / Otsu Thresholding for dark oral cavity
        # Lower 30% intensity is considered oral opening
        mean_val = np.mean(blurred)
        thresh_val = max(35, int(mean_val * 0.65))
        _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # 3. Morphological close to bridge tongue/teeth gaps inside the oral void
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 4. Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_metric = 0.0
        best_cnt = None

        if contours:
            # Filter contours located roughly in central mouth region
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < (w * h * 0.04):  # Ignore tiny specks
                    continue

                cx, cy, cw, ch = cv2.boundingRect(cnt)
                # Check centrality
                cnt_mid_x = cx + cw / 2.0
                if abs(cnt_mid_x - (w / 2.0)) > (w * 0.35):
                    continue

                # Metric: normalized vertical opening height relative to mouth width
                metric = float(ch) / float(w)
                if metric > best_metric:
                    best_metric = metric
                    best_cnt = cnt

        self.last_mouth_metric = best_metric
        self.last_mouth_contour = best_cnt
        return best_metric, best_cnt

    def update(self, face_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes eye presence and mouth opening from face ROIs.
        Updates temporal counters and determines drowsiness and yawning states.
        """
        if face_data is None:
            # Face not detected: clear eye buffer gradually
            self.visibility_buffer.append(0)
            return {
                "eyes_detected": 0,
                "eyes_visible": False,
                "eye_absence_frames": self.eye_absence_frames,
                "visibility_ratio": float(np.mean(self.visibility_buffer)) if self.visibility_buffer else 0.0,
                "is_drowsy": False,
                "total_blinks": self.total_blinks,
                "total_drowsy_events": self.total_drowsy_events,
                "mouth_opening_metric": 0.0,
                "is_yawning": False,
                "total_yawns": self.total_yawns,
                "eye_bboxes": []
            }

        # 1. Eye Visibility Tracking
        eye_roi_rect = face_data["eye_roi_rect"]
        eye_gray = face_data["eye_gray"]
        detected_eyes = self.detect_eyes_in_roi(eye_gray, (eye_roi_rect[0], eye_roi_rect[1]))
        self.last_detected_eyes = detected_eyes

        num_eyes = len(detected_eyes)
        eyes_visible = num_eyes >= 1
        self.visibility_buffer.append(1 if eyes_visible else 0)
        visibility_ratio = float(np.mean(self.visibility_buffer)) if self.visibility_buffer else 0.0

        is_drowsy = False
        if not eyes_visible:
            self.eye_absence_frames += 1
            self.current_blink_duration += 1
            self.blink_in_progress = True

            # Check for prolonged eye closure
            if self.eye_absence_frames >= self.eye_absence_consec:
                is_drowsy = True
                if self.eye_absence_frames == self.eye_absence_consec:
                    self.total_drowsy_events += 1
        else:
            # Eyes visible in this frame
            if self.blink_in_progress:
                # If absence duration was short, count as a normal blink
                if 1 <= self.current_blink_duration <= config.BLINK_MAX_FRAMES:
                    self.total_blinks += 1
                self.blink_in_progress = False
                self.current_blink_duration = 0
            self.eye_absence_frames = 0

        # Heuristic PERCLOS approximation from visibility buffer (if buffer has sufficient frames)
        if len(self.visibility_buffer) >= 30 and visibility_ratio < config.EYE_VISIBILITY_THRESHOLD:
            is_drowsy = True

        # 2. Mouth Yawn Analysis
        mouth_gray = face_data["mouth_gray"]
        mouth_metric, _ = self.analyze_mouth_opening(mouth_gray)

        if self.yawn_cooldown > 0:
            self.yawn_cooldown -= 1

        is_yawning = False
        if mouth_metric >= self.yawn_threshold:
            self.mouth_open_frames += 1
            if self.mouth_open_frames >= self.yawn_consec_frames:
                is_yawning = True
                if not self.is_currently_yawning and self.yawn_cooldown == 0:
                    self.is_currently_yawning = True
                    self.total_yawns += 1
                    self.yawn_cooldown = config.YAWN_COOLDOWN_FRAMES
        else:
            self.mouth_open_frames = 0
            self.is_currently_yawning = False

        return {
            "eyes_detected": num_eyes,
            "eyes_visible": eyes_visible,
            "eye_absence_frames": self.eye_absence_frames,
            "visibility_ratio": float(np.round(visibility_ratio * 100.0, 1)),
            "is_drowsy": is_drowsy,
            "total_blinks": self.total_blinks,
            "total_drowsy_events": self.total_drowsy_events,
            "mouth_opening_metric": float(np.round(mouth_metric, 3)),
            "is_yawning": is_yawning,
            "mouth_open_frames": self.mouth_open_frames,
            "total_yawns": self.total_yawns,
            "eye_bboxes": detected_eyes
        }

    def draw_annotations(self, canvas: np.ndarray,
                         face_data: Optional[Dict[str, Any]],
                         drowsy_data: Dict[str, Any]) -> np.ndarray:
        """Draws eye bounding boxes and mouth cavity outline onto video frame."""
        if face_data is None:
            return canvas

        # 1. Draw detected eye bounding boxes
        for (ex, ey, ew, eh) in drowsy_data.get("eye_bboxes", []):
            cv2.rectangle(canvas, (ex, ey), (ex + ew, ey + eh), (255, 240, 0), 2)
            cv2.putText(canvas, "EYE", (ex, max(10, ey - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 240, 0), 1)

        # 2. Draw mouth detection / opening indicator
        mx, my, mw, mh = face_data["mouth_roi_rect"]
        mom = drowsy_data.get("mouth_opening_metric", 0.0)
        m_color = (0, 100, 255) if drowsy_data.get("is_yawning") else (157, 78, 221)
        cv2.putText(canvas, f"MOUTH OPENING: {mom:.2f}", (mx, my + mh + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, m_color, 1)

        return canvas

    def reset_stats(self):
        """Reset state tracking counters."""
        self.eye_absence_frames = 0
        self.total_blinks = 0
        self.blink_in_progress = False
        self.current_blink_duration = 0
        self.total_drowsy_events = 0
        self.visibility_buffer.clear()
        self.mouth_open_frames = 0
        self.total_yawns = 0
        self.is_currently_yawning = False
        self.yawn_cooldown = 0
