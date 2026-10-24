"""Core package initialization using OpenCV Haar cascades and image processing."""
from core import config
from core.face_detector import FaceDetector
from core.drowsiness_detector import DrowsinessDetector
from core.distraction_detector import DistractionDetector
from core.risk_engine import RiskEngine
from core.alert_manager import AlertManager
from core.stream_manager import StreamManager
