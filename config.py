"""
Configuration settings for Driver Drowsiness and Distraction Detection System.
Uses OpenCV Haar Cascades for face and eye detection, mouth ROI image processing,
and centroid-based head drift approximation. No MediaPipe or neural networks.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database" / "driver_monitor.db"
CASCADES_DIR = BASE_DIR / "cascades"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Haar Cascade File Paths
FACE_CASCADE_PATH = CASCADES_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = CASCADES_DIR / "haarcascade_eye.xml"

# Video Capture Settings
DEFAULT_CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Drowsiness Detection Settings (Eye Tracking via Haar)
# When eyes are not detected in upper face ROI for consecutive frames
EYE_ABSENCE_CONSEC_FRAMES = 12   # Consecutive frames without detected eyes for drowsiness alert (~0.4-0.5s)
BLINK_MAX_FRAMES = 5            # Eye absence of 1 to 5 frames treated as normal blink (~100-180ms)
EYE_VISIBILITY_WINDOW = 60      # Window of frames to estimate eye visibility percentage (~2s)
EYE_VISIBILITY_THRESHOLD = 0.40 # If eyes visible in < 40% of recent window -> fatigue advisory

# Yawning Detection Settings (Mouth ROI Image Processing)
# Threshold on normalized dark cavity opening ratio in lower face ROI
YAWN_OPENING_THRESHOLD = 0.30   # Ratio of detected oral cavity height to mouth width
YAWN_CONSEC_FRAMES = 12         # Consecutive frames of mouth opening for yawn alert
YAWN_COOLDOWN_FRAMES = 45       # Frames to wait before registering next yawn event

# Distraction Detection Settings (Centroid Drift Approximation)
# Normalized drift from image center (0.0 to 1.0)
HEAD_DRIFT_X_THRESHOLD = 0.28   # Normalized horizontal drift (left/right) from frame center
HEAD_DRIFT_DOWN_THRESHOLD = 0.24 # Normalized vertical downward drift (looking down at lap/phone)
DISTRACTION_CONSEC_FRAMES = 16  # Consecutive frames to trigger distraction alert (~0.5-0.6s)

# Risk Assessment Score Weights (0 to 100)
RISK_WEIGHT_EYES = 0.45         # Weight for eye absence / prolonged closure
RISK_WEIGHT_DRIFT = 0.30        # Weight for head position drift (approx distraction)
RISK_WEIGHT_YAWN = 0.25         # Weight for mouth yawn opening

# Risk Thresholds
RISK_LEVEL_NORMAL = 30
RISK_LEVEL_WARNING = 60
RISK_LEVEL_CRITICAL = 85

# Alert Manager Settings
ALERT_COOLDOWN_SECONDS = 2.0
AUDIO_ENABLED_DEFAULT = True
