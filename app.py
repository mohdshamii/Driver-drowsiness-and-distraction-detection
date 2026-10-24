"""
Flask Web Application for Driver Drowsiness and Distraction Detection System.
Provides live webcam MJPEG streaming (cv2.VideoCapture(0)), Haar cascade detection,
REST API endpoints for real-time telemetry, configurable thresholds, and alerts.
"""
import os
import io
import csv
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS
import config
from database.db import init_db, get_recent_alerts, get_all_sessions, get_session_stats
from core.stream_manager import StreamManager

# Initialize Flask app
app = Flask(__name__, template_folder=str(config.TEMPLATES_DIR), static_folder=str(config.STATIC_DIR))
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

# Initialize database
init_db()

# Initialize Stream Manager capturing live webcam feed
stream_mgr = StreamManager(camera_index=config.DEFAULT_CAMERA_INDEX)
stream_mgr.start_camera(camera_index=config.DEFAULT_CAMERA_INDEX)

@app.route("/")
def index():
    """Renders the main driver monitoring cockpit dashboard."""
    return render_template("dashboard.html")

@app.route("/video_feed")
def video_feed():
    """Video streaming route yielding multipart MJPEG stream from live webcam."""
    return Response(
        stream_mgr.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/telemetry")
def get_telemetry():
    """Returns live telemetry snapshot (risk score, eyes detected, mouth metric, head drift)."""
    return jsonify(stream_mgr.latest_telemetry)

@app.route("/api/alerts")
def get_alerts():
    """Returns list of recent safety alerts."""
    limit = request.args.get("limit", default=25, type=int)
    alerts = stream_mgr.alert_manager.get_recent_alerts(limit=limit)
    if not alerts:
        alerts = get_recent_alerts(limit=limit, session_id=stream_mgr.session_id)
    return jsonify({"alerts": alerts, "count": len(alerts)})

@app.route("/api/camera/reconnect", methods=["POST"])
def reconnect_camera():
    """Re-attempts connection to webcam hardware."""
    data = request.get_json() or {}
    cam_idx = data.get("camera_index", stream_mgr.camera_index)
    stream_mgr.start_camera(camera_index=cam_idx)
    return jsonify({
        "status": "SUCCESS" if stream_mgr.is_camera_connected else "FAILED",
        "camera_connected": stream_mgr.is_camera_connected,
        "camera_error": stream_mgr.camera_error
    })

@app.route("/api/session/start", methods=["POST"])
def start_session_route():
    """Starts or switches camera monitoring session."""
    data = request.get_json() or {}
    camera_idx = data.get("camera_index", config.DEFAULT_CAMERA_INDEX)
    file_path = data.get("file_path", None)

    stream_mgr.stop_camera()
    stream_mgr.start_camera(camera_index=camera_idx, file_path=file_path)

    return jsonify({
        "status": "SUCCESS",
        "session_id": stream_mgr.session_id,
        "camera_connected": stream_mgr.is_camera_connected,
        "camera_error": stream_mgr.camera_error
    })

@app.route("/api/session/stop", methods=["POST"])
def stop_session_route():
    """Pauses monitoring and saves session summary."""
    sid = stream_mgr.session_id
    stream_mgr.stop_camera()
    return jsonify({
        "status": "SUCCESS",
        "session_id": sid,
        "message": f"Session #{sid} concluded."
    })

@app.route("/api/session/history")
def session_history():
    """Returns past driving session summaries."""
    sessions = get_all_sessions(limit=30)
    return jsonify({"sessions": sessions})

@app.route("/api/settings", methods=["GET", "POST"])
def handle_settings():
    """Gets or updates threshold parameters."""
    if request.method == "POST":
        data = request.get_json() or {}

        if "eye_absence_consec" in data:
            val = int(data["eye_absence_consec"])
            stream_mgr.drowsiness_detector.eye_absence_consec = val
            config.EYE_ABSENCE_CONSEC_FRAMES = val

        if "yawn_threshold" in data:
            val = float(data["yawn_threshold"])
            stream_mgr.drowsiness_detector.yawn_threshold = val
            config.YAWN_OPENING_THRESHOLD = val

        if "drift_x_thresh" in data:
            val = float(data["drift_x_thresh"])
            stream_mgr.distraction_detector.drift_x_thresh = val
            config.HEAD_DRIFT_X_THRESHOLD = val

        if "audio_enabled" in data:
            enabled = bool(data["audio_enabled"])
            stream_mgr.alert_manager.set_audio_enabled(enabled)

        if "show_face_box" in data:
            stream_mgr.show_face_box = bool(data["show_face_box"])

        if "show_eyes_box" in data:
            stream_mgr.show_eyes_box = bool(data["show_eyes_box"])

        if "show_drift_vector" in data:
            stream_mgr.show_drift_vector = bool(data["show_drift_vector"])

        return jsonify({"status": "SUCCESS", "message": "Thresholds updated successfully."})

    # GET
    return jsonify({
        "eye_absence_consec": stream_mgr.drowsiness_detector.eye_absence_consec,
        "yawn_threshold": stream_mgr.drowsiness_detector.yawn_threshold,
        "drift_x_thresh": stream_mgr.distraction_detector.drift_x_thresh,
        "audio_enabled": stream_mgr.alert_manager.audio_enabled,
        "show_face_box": stream_mgr.show_face_box,
        "show_eyes_box": stream_mgr.show_eyes_box,
        "show_drift_vector": stream_mgr.show_drift_vector,
        "camera_index": stream_mgr.camera_index,
        "camera_connected": stream_mgr.is_camera_connected
    })

@app.route("/api/export/alerts_csv")
def export_alerts_csv():
    """Exports all logged alerts as CSV for data analysis."""
    alerts = get_recent_alerts(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "session_id", "timestamp", "alert_type", "severity", "risk_score", "ear_or_visibility", "mar_or_mouth", "drift_x", "drift_y", "message"])
    for a in alerts:
        writer.writerow([
            a.get("id"), a.get("session_id"), a.get("timestamp"),
            a.get("alert_type"), a.get("severity"), a.get("risk_score"),
            a.get("ear"), a.get("mar"), a.get("yaw"), a.get("pitch"),
            a.get("message")
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=driver_alerts_export.csv"}
    )

if __name__ == "__main__":
    print("=" * 60)
    print("Driver Drowsiness & Distraction Detection System")
    print("Capturing live video from cv2.VideoCapture(0) with OpenCV Haar Cascades")
    print("Dashboard available at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
