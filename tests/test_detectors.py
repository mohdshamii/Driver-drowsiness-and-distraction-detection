"""
Unit and Integration Tests for OpenCV Haar Cascade Driver Drowsiness and Distraction Detection System.
Tests face cascade, eye visibility tracker, mouth image processing, centroid drift heuristic,
risk engine, and camera error handling.
"""
import unittest
import os
import cv2
import numpy as np
import config
from database.db import init_db
from core.face_detector import FaceDetector
from core.drowsiness_detector import DrowsinessDetector
from core.distraction_detector import DistractionDetector
from core.risk_engine import RiskEngine
from core.alert_manager import AlertManager
from core.stream_manager import StreamManager

class TestHaarDriverMonitoringSystem(unittest.TestCase):
    def setUp(self):
        init_db()
        self.face_detector = FaceDetector()
        self.drowsy_detector = DrowsinessDetector()
        self.distract_detector = DistractionDetector()
        self.risk_engine = RiskEngine()
        self.alert_manager = AlertManager()

    def test_haar_cascades_exist_and_load(self):
        """Verify that official OpenCV Haar Cascade XML files exist and are loaded."""
        self.assertTrue(os.path.exists(config.FACE_CASCADE_PATH), "Face cascade XML missing.")
        self.assertTrue(os.path.exists(config.EYE_CASCADE_PATH), "Eye cascade XML missing.")
        self.assertFalse(self.face_detector.face_cascade.empty(), "Face cascade failed to initialize.")
        self.assertFalse(self.drowsy_detector.eye_cascade.empty(), "Eye cascade failed to initialize.")

    def test_drowsiness_eye_absence_and_blinking(self):
        """
        Verify that short eye absence (1-3 frames) counts as a blink,
        while prolonged eye absence (>=12 frames) triggers drowsiness.
        """
        detector = DrowsinessDetector(eye_absence_consec=12)

        # 1. Simulate Normal Blink (2 frames of missing eyes)
        # Dummy face data with blank eye ROI
        mock_face_data = {
            "bbox": (100, 100, 200, 200),
            "centroid": (200, 200),
            "norm_pos": (0.0, 0.0),
            "eye_roi_rect": (116, 136, 168, 80),
            "mouth_roi_rect": (136, 224, 128, 66),
            "eye_gray": np.zeros((80, 168), dtype=np.uint8), # No eyes detectable in blank image
            "mouth_gray": np.zeros((66, 128), dtype=np.uint8)
        }

        # Frame 1 & 2: Eyes closed (blink in progress)
        res1 = detector.update(mock_face_data)
        self.assertFalse(res1["eyes_visible"])
        self.assertFalse(res1["is_drowsy"])

        res2 = detector.update(mock_face_data)
        self.assertFalse(res2["is_drowsy"])

        # Frame 3: Eyes reopen (simulate by passing mock eye detection)
        detector.detect_eyes_in_roi = lambda roi, offset: [(130, 150, 30, 25), (190, 150, 30, 25)]
        res3 = detector.update(mock_face_data)
        self.assertTrue(res3["eyes_visible"])
        self.assertEqual(res3["total_blinks"], 1, "Completed 2-frame blink should increment total_blinks.")
        self.assertFalse(res3["is_drowsy"])

        # 2. Simulate Prolonged Eye Closure (12 consecutive frames without eyes)
        detector.detect_eyes_in_roi = lambda roi, offset: [] # No eyes
        for _ in range(11):
            res = detector.update(mock_face_data)
            self.assertFalse(res["is_drowsy"])

        # 12th frame: Prolonged closure reached
        res12 = detector.update(mock_face_data)
        self.assertTrue(res12["is_drowsy"], "Eyes closed for >=12 consecutive frames must trigger is_drowsy.")
        self.assertEqual(res12["total_drowsy_events"], 1)

    def test_mouth_yawn_opening_analysis(self):
        """Verify mouth cavity thresholding and yawn trigger."""
        detector = DrowsinessDetector(yawn_threshold=0.30, yawn_consec_frames=5)

        # 1. Closed mouth: uniformly bright skin ROI with no large dark cavity
        closed_mouth = np.full((70, 130), 160, dtype=np.uint8)
        metric_closed, _ = detector.analyze_mouth_opening(closed_mouth)
        self.assertLess(metric_closed, 0.20, "Closed mouth should produce small mouth opening metric.")

        # 2. Open yawn mouth: dark oral void ellipse in center
        yawn_mouth = np.full((70, 130), 160, dtype=np.uint8)
        cv2.ellipse(yawn_mouth, (65, 35), (28, 24), 0, 0, 360, 25, -1) # Dark cavity (intensity 25)
        metric_yawn, cnt = detector.analyze_mouth_opening(yawn_mouth)
        self.assertGreater(metric_yawn, 0.30, f"Yawning mouth opening metric should exceed 0.30, got {metric_yawn}")
        self.assertIsNotNone(cnt)

    def test_distraction_centroid_drift_approximation(self):
        """
        Verify that head centroid offset from calibrated frame center
        detects distraction and is explicitly labeled as heuristic approximation.
        """
        distract = DistractionDetector(drift_x_thresh=0.28, drift_down_thresh=0.24, consec_frames=5)

        # 1. Centered face (norm_pos = 0.0, 0.0): Attentive
        centered_face = {
            "centroid": (320, 240),
            "norm_pos": (0.0, 0.0)
        }
        res_normal = distract.update(centered_face, (640, 480))
        self.assertFalse(res_normal["is_distracted"])
        self.assertIn("Heuristic Approx", res_normal["label"])

        # 2. Shifted face looking right: norm_pos = (+0.40, 0.0) for 5 frames
        drifted_face = {
            "centroid": (448, 240),
            "norm_pos": (0.40, 0.0)
        }
        for _ in range(4):
            res = distract.update(drifted_face, (640, 480), smooth=False)
            self.assertFalse(res["is_distracted"])

        # 5th frame: reaches threshold
        res_drift = distract.update(drifted_face, (640, 480), smooth=False)
        self.assertTrue(res_drift["is_distracted"])
        self.assertEqual(res_drift["distraction_type"], "LOOKING_AWAY_RIGHT")
        self.assertIn("Heuristic Approx", res_drift["label"])

    def test_risk_engine_fused_score(self):
        """Verify normal driving state risk score is low and danger state risk score is high."""
        # Normal attentive state
        normal_drowsy = {"eye_absence_frames": 0, "visibility_ratio": 100.0, "mouth_opening_metric": 0.08, "mouth_open_frames": 0}
        normal_distract = {"distraction_frames": 0, "distraction_type": "NONE"}
        risk_normal = self.risk_engine.evaluate(normal_drowsy, normal_distract, smooth=False)
        self.assertLess(risk_normal["risk_score"], 30.0)
        self.assertEqual(risk_normal["status_level"], "NORMAL")

        # Dangerous state (eyes closed for 24 frames, head tilted down looking at phone)
        danger_drowsy = {"eye_absence_frames": 24, "visibility_ratio": 10.0, "mouth_opening_metric": 0.12, "mouth_open_frames": 0, "is_drowsy": True}
        danger_distract = {"distraction_frames": 20, "distraction_type": "LOOKING_DOWN_PHONE", "is_distracted": True}
        risk_danger = self.risk_engine.evaluate(danger_drowsy, danger_distract, smooth=False)
        self.assertGreater(risk_danger["risk_score"], 60.0)
        self.assertEqual(risk_danger["status_level"], "CRITICAL")

    def test_stream_manager_camera_handling(self):
        """Verify StreamManager handles camera initialization and diagnostic error frames gracefully."""
        sm = StreamManager(camera_index=999) # Non-existent camera index to test error handling
        sm.start_camera(camera_index=999)

        # Should generate diagnostic error frame without crashing
        canvas, telemetry = sm.get_processed_frame()
        self.assertEqual(canvas.shape, (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3))
        self.assertFalse(telemetry["camera_connected"])
        self.assertIsNotNone(telemetry["camera_error"])

        sm.stop_camera()
        self.assertFalse(sm.is_running)

if __name__ == "__main__":
    unittest.main()
