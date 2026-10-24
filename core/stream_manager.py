"""
Stream Manager Module.
Coordinates real-time webcam acquisition via cv2.VideoCapture(0),
applies OpenCV Haar Cascade face detection, eye visibility tracking,
mouth ROI yawning image processing, and centroid drift distraction estimation.
No MediaPipe, no neural networks, no simulation.
"""
import time
import os
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Any
import config
from core.face_detector import FaceDetector
from core.drowsiness_detector import DrowsinessDetector
from core.distraction_detector import DistractionDetector
from core.risk_engine import RiskEngine
from core.alert_manager import AlertManager
from database.db import create_session, end_session, log_telemetry_batch

class StreamManager:
    def __init__(self, camera_index: int = config.DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.camera_error = None
        self.is_camera_connected = False

        # OpenCV Haar & Heuristic Subsystems
        self.face_detector = FaceDetector()
        self.drowsiness_detector = DrowsinessDetector()
        self.distraction_detector = DistractionDetector()
        self.risk_engine = RiskEngine()
        self.alert_manager = AlertManager()

        # Session tracking
        self.session_id = None
        self.session_start_time = 0.0
        self.total_frames_processed = 0

        # UI Overlay Settings
        self.show_face_box = True
        self.show_eyes_box = True
        self.show_drift_vector = True
        self.show_telemetry_hud = True

        # Telemetry batch buffer for database
        self.telemetry_batch = []
        self.last_db_flush_time = time.time()

        # FPS calculation
        self.prev_frame_time = time.time()
        self.fps = 0.0

        # Latest telemetry cache for REST API
        self.latest_telemetry = {
            "status": "STOPPED",
            "status_level": "NORMAL",
            "risk_score": 0.0,
            "max_risk_score": 0.0,
            "face_detected": False,
            "eyes_detected": 0,
            "eyes_visible": False,
            "eye_absence_frames": 0,
            "visibility_ratio": 100.0,
            "mouth_opening_metric": 0.0,
            "is_yawning": False,
            "norm_drift_x": 0.0,
            "norm_drift_y": 0.0,
            "movement_speed": 0.0,
            "is_distracted": False,
            "distraction_type": "NONE",
            "distraction_label": "DISTRACTION (Heuristic Approx: Head Drift/Turn)",
            "blinks": 0,
            "yawns": 0,
            "drowsiness_alerts": 0,
            "distraction_alerts": 0,
            "fps": 0.0,
            "camera_connected": False,
            "camera_error": None
        }

    def start_camera(self, camera_index: Optional[int] = None, file_path: Optional[str] = None):
        """
        Initializes video capture from actual webcam device (or video file).
        Handles permissions, busy device locks, and connection errors.
        """
        if camera_index is not None:
            self.camera_index = camera_index

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.camera_error = None
        self.is_camera_connected = False

        try:
            if file_path:
                print(f"[StreamManager] Opening video file: {file_path}")
                self.cap = cv2.VideoCapture(file_path)
            else:
                print(f"[StreamManager] Connecting to live webcam cv2.VideoCapture({self.camera_index})...")
                # On Windows, try cv2.CAP_DSHOW for rapid hardware acquisition; fallback to default
                if os.name == 'nt':
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.camera_index)
                else:
                    self.cap = cv2.VideoCapture(self.camera_index)

                # Set desired resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            if not self.cap.isOpened():
                self.camera_error = f"Cannot open webcam (device index {self.camera_index}). Check camera connection or privacy permissions."
                print(f"[StreamManager] ERROR: {self.camera_error}")
                self.is_camera_connected = False
            else:
                # Test read one frame
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    self.camera_error = f"Webcam index {self.camera_index} opened but returned no video frames (device may be busy or shutter closed)."
                    print(f"[StreamManager] ERROR: {self.camera_error}")
                    self.is_camera_connected = False
                else:
                    self.is_camera_connected = True
                    print(f"[StreamManager] Webcam successfully streaming! Frame resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")

        except Exception as e:
            self.camera_error = f"Camera initialization exception: {str(e)}"
            print(f"[StreamManager] EXCEPTION: {self.camera_error}")
            self.is_camera_connected = False

        # Start database monitoring session
        self.session_id = create_session()
        self.session_start_time = time.time()
        self.total_frames_processed = 0
        self.is_running = True

    def stop_camera(self):
        """Stops webcam capture and finalizes driving session."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_camera_connected = False

        if self.session_id is not None:
            duration = time.time() - self.session_start_time
            summary = {
                "duration_seconds": duration,
                "total_blinks": self.drowsiness_detector.total_blinks,
                "total_yawns": self.drowsiness_detector.total_yawns,
                "drowsiness_events": self.drowsiness_detector.total_drowsy_events,
                "distraction_events": self.distraction_detector.total_distraction_events,
                "max_risk_score": self.risk_engine.max_session_risk,
                "avg_ear": float(np.mean(self.drowsiness_detector.visibility_buffer)) if self.drowsiness_detector.visibility_buffer else 0.0,
                "avg_mar": float(self.drowsiness_detector.last_mouth_metric)
            }
            end_session(self.session_id, summary)
            print(f"[StreamManager] Session #{self.session_id} finalized.")
            self.session_id = None

    def generate_error_frame(self) -> np.ndarray:
        """
        Renders a diagnostic error screen when webcam cannot be accessed.
        Provides helpful hardware and privacy permission guidance.
        """
        w, h = config.FRAME_WIDTH, config.FRAME_HEIGHT
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark grid background
        canvas[:] = (14, 18, 28)
        for y in range(0, h, 30):
            cv2.line(canvas, (0, y), (w, y), (22, 28, 42), 1)
        for x in range(0, w, 30):
            cv2.line(canvas, (x, 0), (x, h), (22, 28, 42), 1)

        # Warning Card Box
        cv2.rectangle(canvas, (30, 40), (w - 30, h - 40), (20, 26, 40), -1)
        cv2.rectangle(canvas, (30, 40), (w - 30, h - 40), (0, 70, 255), 2)

        # Warning Icon & Title
        cv2.putText(canvas, "! CAMERA HARDWARE ACCESS ERROR !", (50, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 70, 255), 2)

        err_msg = self.camera_error or f"Unable to read live video feed from cv2.VideoCapture({self.camera_index})"
        cv2.putText(canvas, err_msg[:56], (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1)

        # Troubleshooting Steps
        steps = [
            "Troubleshooting Steps:",
            "1. Connect an external or built-in webcam to your system.",
            "2. Ensure physical camera shutter or privacy switch is OPEN.",
            "3. Close other apps using camera (Zoom, Teams, Discord, Browser).",
            "4. Check Windows Settings -> Privacy & Security -> Camera = ON.",
            "5. Try selecting Camera Device 1 from the toolbar dropdown."
        ]

        y_offset = 160
        for step in steps:
            color = (0, 240, 255) if step.startswith("Troubleshooting") else (170, 185, 205)
            cv2.putText(canvas, step, (50, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)
            y_offset += 26

        # Retry status
        pulse = "WAITING FOR WEBCAM FEED..." if (int(time.time() * 2) % 2 == 0) else "PLEASE PLUG IN WEBCAM"
        cv2.putText(canvas, pulse, (50, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 170), 1)

        return canvas

    def get_processed_frame(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Captures live webcam frame, runs OpenCV Haar face & eye detection,
        mouth yawning analysis, and head centroid drift estimation.
        Annotates HUD and updates telemetry.
        """
        raw_frame = None

        if not self.is_camera_connected or self.cap is None:
            # Reconnect attempt periodically
            if int(time.time()) % 3 == 0:
                self.start_camera(self.camera_index)

            if not self.is_camera_connected:
                raw_frame = self.generate_error_frame()
                self.latest_telemetry["camera_connected"] = False
                self.latest_telemetry["camera_error"] = self.camera_error
                return raw_frame, self.latest_telemetry
        else:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.is_camera_connected = False
                self.camera_error = "Lost connection to live camera stream."
                raw_frame = self.generate_error_frame()
                return raw_frame, self.latest_telemetry
            raw_frame = frame

        # Standardize resolution
        if raw_frame.shape[1] != config.FRAME_WIDTH or raw_frame.shape[0] != config.FRAME_HEIGHT:
            raw_frame = cv2.resize(raw_frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # FPS Tracking
        now = time.time()
        dt = now - self.prev_frame_time
        self.fps = (1.0 / dt) if dt > 0 else 30.0
        self.prev_frame_time = now
        self.total_frames_processed += 1

        canvas = raw_frame.copy()

        # 1. OpenCV Haar Face Detection
        face_data = self.face_detector.process_frame(raw_frame)
        face_detected = face_data is not None

        # 2. Eye Tracking & Drowsiness via Haar Eye Cascade
        drowsy_data = self.drowsiness_detector.update(face_data)

        # 3. Head Position Drift & Distraction Approximation
        distract_data = self.distraction_detector.update(face_data, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        # 4. Multi-Factor Risk Assessment
        risk_data = self.risk_engine.evaluate(drowsy_data, distract_data)

        # 5. Alert Manager & Cooldown
        alert = self.alert_manager.process_state(
            session_id=self.session_id,
            drowsy_data=drowsy_data,
            distract_data=distract_data,
            risk_data=risk_data
        )

        # 6. Render HUD Annotations onto Video Frame
        if self.show_face_box:
            canvas = self.face_detector.draw_face_annotations(canvas, face_data)
        if self.show_eyes_box:
            canvas = self.drowsiness_detector.draw_annotations(canvas, face_data, drowsy_data)
        if self.show_drift_vector:
            canvas = self.distraction_detector.draw_annotations(canvas, face_data, distract_data)

        if self.show_telemetry_hud:
            canvas = self._draw_hud_overlay(canvas, face_data, drowsy_data, distract_data, risk_data, alert)

        # 7. Record Telemetry to Database Batch
        if self.session_id and (now - self.last_db_flush_time > 0.5):
            self.telemetry_batch.append((
                self.session_id,
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                drowsy_data.get("visibility_ratio", 100.0),
                drowsy_data.get("mouth_opening_metric", 0.0),
                float(drowsy_data.get("eyes_detected", 0)),
                distract_data.get("norm_drift_x", 0.0),
                distract_data.get("norm_drift_y", 0.0),
                risk_data.get("risk_score", 0.0),
                risk_data.get("status", "NORMAL")
            ))
            if len(self.telemetry_batch) >= 10:
                log_telemetry_batch(self.telemetry_batch)
                self.telemetry_batch.clear()
            self.last_db_flush_time = now

        # Update cached latest telemetry for REST API
        self.latest_telemetry = {
            "status": risk_data.get("status", "NORMAL"),
            "status_level": risk_data.get("status_level", "NORMAL"),
            "risk_score": risk_data.get("risk_score", 0.0),
            "max_risk_score": risk_data.get("max_risk_score", 0.0),
            "face_detected": face_detected,
            "eyes_detected": drowsy_data.get("eyes_detected", 0),
            "eyes_visible": drowsy_data.get("eyes_visible", False),
            "eye_absence_frames": drowsy_data.get("eye_absence_frames", 0),
            "visibility_ratio": drowsy_data.get("visibility_ratio", 100.0),
            "mouth_opening_metric": drowsy_data.get("mouth_opening_metric", 0.0),
            "is_yawning": drowsy_data.get("is_yawning", False),
            "norm_drift_x": distract_data.get("norm_drift_x", 0.0),
            "norm_drift_y": distract_data.get("norm_drift_y", 0.0),
            "movement_speed": distract_data.get("movement_speed", 0.0),
            "is_distracted": distract_data.get("is_distracted", False),
            "distraction_type": distract_data.get("distraction_type", "NONE"),
            "distraction_label": "DISTRACTION (Heuristic Approx: Head Drift/Turn)",
            "blinks": drowsy_data.get("total_blinks", 0),
            "yawns": drowsy_data.get("total_yawns", 0),
            "drowsiness_alerts": drowsy_data.get("total_drowsy_events", 0),
            "distraction_alerts": distract_data.get("total_distractions", 0),
            "fps": float(np.round(self.fps, 1)),
            "active_alert": alert,
            "camera_connected": self.is_camera_connected,
            "camera_error": self.camera_error
        }

        return canvas, self.latest_telemetry

    def _draw_hud_overlay(self, canvas: np.ndarray,
                          face_data: Optional[Dict[str, Any]],
                          drowsy_data: Dict[str, Any],
                          distract_data: Dict[str, Any],
                          risk_data: Dict[str, Any],
                          alert: Optional[Dict[str, Any]]) -> np.ndarray:
        """Draws top status pill, risk meter, bottom telemetry bar, and alert banners."""
        h, w, _ = canvas.shape
        status_level = risk_data.get("status_level", "NORMAL")
        status_text = risk_data.get("status", "ALERT")
        risk_score = risk_data.get("risk_score", 0.0)

        # Palette
        if status_level == "CRITICAL":
            theme_color = (0, 0, 255)
            bg_badge = (20, 20, 60)
        elif status_level == "WARNING":
            theme_color = (0, 180, 255)
            bg_badge = (20, 40, 50)
        else:
            theme_color = (0, 255, 170)
            bg_badge = (15, 30, 25)

        # 1. Top Status Badge
        badge_w, badge_h = 260, 36
        badge_x, badge_y = 15, 15
        overlay = canvas.copy()
        cv2.rectangle(overlay, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), bg_badge, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), theme_color, 2)

        pulse_r = 7 if (int(time.time() * 3) % 2 == 0) else 5
        cv2.circle(canvas, (badge_x + 18, badge_y + badge_h // 2), pulse_r, theme_color, -1)
        cv2.putText(canvas, f"STATUS: {status_text[:18]}", (badge_x + 32, badge_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        # 2. Risk Meter Bar (Top Right)
        meter_w, meter_h = 160, 36
        meter_x = w - meter_w - 15
        meter_y = 15
        overlay = canvas.copy()
        cv2.rectangle(overlay, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (15, 20, 30), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (meter_x, meter_y), (meter_x + meter_w, meter_y + meter_h), (80, 90, 110), 1)

        fill_w = int((meter_w - 8) * (risk_score / 100.0))
        cv2.rectangle(canvas, (meter_x + 4, meter_y + 4), (meter_x + 4 + fill_w, meter_y + meter_h - 4), theme_color, -1)
        cv2.putText(canvas, f"RISK: {risk_score:.0f}%", (meter_x + 38, meter_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        # 3. Bottom Telemetry Bar
        bar_h = 32
        bar_y = h - bar_h - 10
        overlay = canvas.copy()
        cv2.rectangle(overlay, (15, bar_y), (w - 15, bar_y + bar_h), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
        cv2.rectangle(canvas, (15, bar_y), (w - 15, bar_y + bar_h), (0, 240, 255), 1)

        num_eyes = drowsy_data.get("eyes_detected", 0)
        vis_pct = drowsy_data.get("visibility_ratio", 100.0)
        mom = drowsy_data.get("mouth_opening_metric", 0.0)
        drift_x = distract_data.get("norm_drift_x", 0.0)

        telemetry_str = f"EYES:{num_eyes}  VIS:{vis_pct:.0f}%  MOUTH:{mom:.2f}  DRIFT:{drift_x:+.2f}  FPS:{self.fps:.0f}"
        cv2.putText(canvas, telemetry_str, (25, bar_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 255), 1)

        # 4. Critical Warning Banner
        if alert or status_level == "CRITICAL":
            banner_y = h // 2 - 30
            overlay = canvas.copy()
            cv2.rectangle(overlay, (20, banner_y), (w - 20, banner_y + 60), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
            cv2.rectangle(canvas, (20, banner_y), (w - 20, banner_y + 60), (0, 0, 255), 3)

            alert_msg = alert["message"] if alert else f"WARNING: {status_text}!"
            cv2.putText(canvas, "! DRIVER ATTENTION ALARM !", (40, banner_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            cv2.putText(canvas, alert_msg[:52], (40, banner_y + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 240, 255), 1)

        return canvas

    def generate_mjpeg_stream(self):
        """Yields MJPEG encoded video frames for Flask streaming response."""
        while self.is_running:
            canvas, _ = self.get_processed_frame()
            ret, jpeg = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ret:
                continue
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1.0 / config.TARGET_FPS)
