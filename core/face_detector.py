"""
Face Detector Module using OpenCV Haar Cascade (haarcascade_frontalface_default.xml).
Detects faces in live webcam frames, extracts facial bounding boxes,
upper-face eye regions, and lower-face mouth regions. No MediaPipe or neural networks.
"""
import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any
import config

class FaceDetector:
    def __init__(self, cascade_path: Optional[str] = None):
        self.cascade_path = str(cascade_path or config.FACE_CASCADE_PATH)
        if not os.path.exists(self.cascade_path):
            raise FileNotFoundError(f"Haar Cascade XML file not found at: {self.cascade_path}")
        
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from: {self.cascade_path}")

        self.last_face_bbox = None
        self.no_face_frames = 0

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detects primary human face in frame using Haar Cascade.
        Returns face bounding box, centroid, and sub-ROIs for eye & mouth processing.
        """
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply gentle histogram equalization or contrast enhancement for lighting robustness
        equalized_gray = cv2.equalizeHist(gray)

        # Detect faces with tuned parameters
        faces = self.face_cascade.detectMultiScale(
            equalized_gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(70, 70),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            self.no_face_frames += 1
            return None

        self.no_face_frames = 0

        # Select largest detected face (closest driver)
        primary_face = max(faces, key=lambda b: b[2] * b[3])
        fx, fy, fw, fh = primary_face
        self.last_face_bbox = primary_face

        # Centroid coordinates
        cx = int(fx + fw / 2)
        cy = int(fy + fh / 2)

        # Normalized coordinates relative to frame center (range -1.0 to +1.0)
        norm_cx = (cx - (w / 2.0)) / (w / 2.0)
        norm_cy = (cy - (h / 2.0)) / (h / 2.0)

        # 1. Eye Region of Interest (Upper half of face)
        eye_y = int(fy + fh * 0.18)
        eye_h = int(fh * 0.40)
        eye_x = int(fx + fw * 0.08)
        eye_w = int(fw * 0.84)
        eye_roi_rect = (eye_x, eye_y, eye_w, eye_h)

        # 2. Mouth Region of Interest (Lower third of face)
        mouth_y = int(fy + fh * 0.62)
        mouth_h = int(fh * 0.33)
        mouth_x = int(fx + fw * 0.18)
        mouth_w = int(fw * 0.64)
        mouth_roi_rect = (mouth_x, mouth_y, mouth_w, mouth_h)

        # Extract grayscale ROIs
        face_gray = gray[fy:fy+fh, fx:fx+fw]
        eye_gray = gray[max(0, eye_y):min(h, eye_y+eye_h), max(0, eye_x):min(w, eye_x+eye_w)]
        mouth_gray = gray[max(0, mouth_y):min(h, mouth_y+mouth_h), max(0, mouth_x):min(w, mouth_x+mouth_w)]

        return {
            "bbox": (int(fx), int(fy), int(fw), int(fh)),
            "centroid": (cx, cy),
            "norm_pos": (float(norm_cx), float(norm_cy)),
            "face_gray": face_gray,
            "eye_roi_rect": eye_roi_rect,
            "mouth_roi_rect": mouth_roi_rect,
            "eye_gray": eye_gray,
            "mouth_gray": mouth_gray,
            "frame_dims": (w, h)
        }

    def draw_face_annotations(self, canvas: np.ndarray, face_data: Optional[Dict[str, Any]]) -> np.ndarray:
        """Draws aesthetic HUD bounding box and detection status on video frame."""
        h, w, _ = canvas.shape

        if face_data is None:
            # Draw subtle searching reticle when face is absent
            cx, cy = w // 2, h // 2
            color_warn = (0, 165, 255)
            cv2.circle(canvas, (cx, cy), 45, color_warn, 1)
            cv2.line(canvas, (cx - 60, cy), (cx + 60, cy), color_warn, 1)
            cv2.line(canvas, (cx, cy - 60), (cx, cy + 60), color_warn, 1)
            cv2.putText(canvas, "! NO FACE DETECTED (SEARCHING...) !", (cx - 150, cy + 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_warn, 2)
            return canvas

        fx, fy, fw, fh = face_data["bbox"]
        cx, cy = face_data["centroid"]

        # Corner bracket length
        c_len = min(22, fw // 4, fh // 4)
        hud_color = (0, 255, 170) # Neon green/cyan

        # Top-left corner
        cv2.line(canvas, (fx, fy), (fx + c_len, fy), hud_color, 2)
        cv2.line(canvas, (fx, fy), (fx, fy + c_len), hud_color, 2)
        # Top-right corner
        cv2.line(canvas, (fx + fw, fy), (fx + fw - c_len, fy), hud_color, 2)
        cv2.line(canvas, (fx + fw, fy), (fx + fw, fy + c_len), hud_color, 2)
        # Bottom-left corner
        cv2.line(canvas, (fx, fy + fh), (fx + c_len, fy + fh), hud_color, 2)
        cv2.line(canvas, (fx, fy + fh), (fx, fy + fh - c_len), hud_color, 2)
        # Bottom-right corner
        cv2.line(canvas, (fx + fw, fy + fh), (fx + fw - c_len, fy + fh), hud_color, 2)
        cv2.line(canvas, (fx + fw, fy + fh), (fx + fw, fy + fh - c_len), hud_color, 2)

        # Centroid target crosshair
        cv2.drawMarker(canvas, (cx, cy), (0, 240, 255), cv2.MARKER_CROSS, 14, 1)

        # Face Label Badge above bounding box
        label_text = f"HAAR FACE: {fw}x{fh}px"
        cv2.rectangle(canvas, (fx, max(0, fy - 22)), (fx + 160, fy), (15, 25, 35), -1)
        cv2.putText(canvas, label_text, (fx + 6, max(12, fy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, hud_color, 1)

        # Draw Eye and Mouth ROI boundary boxes (dashed/subtle)
        ex, ey, ew, eh = face_data["eye_roi_rect"]
        cv2.rectangle(canvas, (ex, ey), (ex + ew, ey + eh), (255, 200, 0), 1)

        mx, my, mw, mh = face_data["mouth_roi_rect"]
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (157, 78, 221), 1)

        return canvas
