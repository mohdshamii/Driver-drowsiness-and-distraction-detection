"""
Alert Manager Module.
Controls alerting policy, cooling-off periods, database logging,
and frontend audio/visual alarm triggers.
"""
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any
import config
from database.db import log_alert

class AlertManager:
    def __init__(self,
                 cooldown_seconds: float = config.ALERT_COOLDOWN_SECONDS,
                 audio_enabled: bool = config.AUDIO_ENABLED_DEFAULT):
        self.cooldown_seconds = cooldown_seconds
        self.audio_enabled = audio_enabled

        # Track timestamps of last alert per type
        self.last_alert_times = {
            "DROWSINESS": 0.0,
            "YAWNING": 0.0,
            "DISTRACTION": 0.0,
            "CRITICAL": 0.0
        }

        # Active alarm trigger state
        self.active_alarm = False
        self.active_alarm_type = None
        self.active_alarm_message = ""

        # Ring buffer of recent alerts
        self.recent_alerts = deque(maxlen=50)

    def process_state(self,
                      session_id: Optional[int],
                      drowsy_data: Dict[str, Any],
                      distract_data: Dict[str, Any],
                      risk_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluates risk state and triggers alerts with cooldown rules.
        """
        now = time.time()
        risk_score = risk_data.get("risk_score", 0.0)
        status_level = risk_data.get("status_level", "NORMAL")
        status = risk_data.get("status", "ALERT")

        alert_type = None
        severity = "WARNING"
        message = ""

        # 1. Critical compound risk or prolonged drowsiness
        if status_level == "CRITICAL":
            severity = "DANGER"
            if drowsy_data.get("is_drowsy") and distract_data.get("is_distracted"):
                alert_type = "CRITICAL"
                message = "CRITICAL WARNING! Prolonged eye closure & distraction detected!"
            elif drowsy_data.get("is_drowsy"):
                alert_type = "DROWSINESS"
                absence_sec = drowsy_data.get("eye_absence_frames", 0) / 30.0
                message = f"DROWSINESS ALERT! Eyes not visible for {absence_sec:.1f}s. Pull over safely!"
            elif distract_data.get("is_distracted"):
                alert_type = "DISTRACTION"
                dtype = distract_data.get("distraction_type", "LOOKING AWAY")
                message = f"DISTRACTION ALERT! Sustained {dtype} (Heuristic Approx). Keep eyes on the road!"
            else:
                alert_type = "CRITICAL"
                message = f"HIGH RISK LEVEL! Driver fatigue score: {risk_score}%"

        # 2. Warning level alerts
        elif status_level == "WARNING":
            severity = "WARNING"
            if drowsy_data.get("is_yawning"):
                alert_type = "YAWNING"
                yawn_count = drowsy_data.get("total_yawns", 1)
                message = f"FATIGUE ADVISORY! Yawn detected (#{yawn_count}). Consider taking a rest break."
            elif distract_data.get("is_distracted"):
                alert_type = "DISTRACTION"
                message = "DISTRACTION DETECTED! Head position drifting away from road center."
            elif drowsy_data.get("is_drowsy"):
                alert_type = "DROWSINESS"
                message = "DROWSINESS DETECTED! Low eye visibility rate."

        # Check if alert condition met and cooldown expired
        if alert_type:
            time_since_last = now - self.last_alert_times.get(alert_type, 0.0)
            if time_since_last >= self.cooldown_seconds:
                self.last_alert_times[alert_type] = now
                self.active_alarm = True
                self.active_alarm_type = alert_type
                self.active_alarm_message = message

                vis_ratio = drowsy_data.get("visibility_ratio", 100.0)
                mom = drowsy_data.get("mouth_opening_metric", 0.0)
                drift_x = distract_data.get("norm_drift_x", 0.0)
                drift_y = distract_data.get("norm_drift_y", 0.0)

                alert_event = {
                    "id": len(self.recent_alerts) + 1,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "alert_type": alert_type,
                    "severity": severity,
                    "risk_score": risk_score,
                    "message": message,
                    "audio_trigger": self.audio_enabled,
                    "ear": vis_ratio,
                    "mar": mom,
                    "yaw": drift_x,
                    "pitch": drift_y
                }

                self.recent_alerts.appendleft(alert_event)

                # Persist to SQLite
                log_alert(
                    session_id=session_id,
                    alert_type=alert_type,
                    severity=severity,
                    risk_score=risk_score,
                    ear=vis_ratio,
                    mar=mom,
                    yaw=drift_x,
                    pitch=drift_y,
                    message=message
                )

                return alert_event
        else:
            # Clear active alarm when returned to normal
            if status_level == "NORMAL":
                self.active_alarm = False
                self.active_alarm_type = None
                self.active_alarm_message = ""

        return None

    def get_recent_alerts(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Return list of recent alerts."""
        return list(self.recent_alerts)[:limit]

    def set_audio_enabled(self, enabled: bool):
        """Toggle audio alerts."""
        self.audio_enabled = enabled
