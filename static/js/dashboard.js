/**
 * Driver Guardian - Modern Enterprise Dashboard Controller & Web Audio Engine
 * Pure OpenCV Haar Cascade telemetry updates, audio alerts, and controls.
 * Adheres to enterprise theme with clear telemetry and zero emojis.
 */

// Enterprise Web Audio Synthesizer
class CockpitSoundEngine {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.volume = 0.8;
    this.lastSoundTime = 0;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  playChime() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    if (now - this.lastSoundTime < 1.0) return;
    this.lastSoundTime = now;

    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gainNode = this.ctx.createGain();

    osc1.type = 'sine';
    osc2.type = 'triangle';

    osc1.frequency.setValueAtTime(587.33, now); // D5
    osc1.frequency.exponentialRampToValueAtTime(880.0, now + 0.15); // A5

    osc2.frequency.setValueAtTime(440.0, now);
    osc2.frequency.exponentialRampToValueAtTime(659.25, now + 0.15);

    gainNode.gain.setValueAtTime(0.01, now);
    gainNode.gain.linearRampToValueAtTime(this.volume * 0.22, now + 0.04);
    gainNode.gain.exponentialRampToValueAtTime(0.001, now + 0.45);

    osc1.connect(gainNode);
    osc2.connect(gainNode);
    gainNode.connect(this.ctx.destination);

    osc1.start(now);
    osc2.start(now);
    osc1.stop(now + 0.5);
    osc2.stop(now + 0.5);
  }

  playUrgentAlarm() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;

    const now = this.ctx.currentTime;
    if (now - this.lastSoundTime < 0.6) return;
    this.lastSoundTime = now;

    for (let i = 0; i < 2; i++) {
      const pulseTime = now + (i * 0.18);
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(950, pulseTime);
      osc.frequency.setValueAtTime(1250, pulseTime + 0.08);

      gain.gain.setValueAtTime(0.01, pulseTime);
      gain.gain.linearRampToValueAtTime(this.volume * 0.32, pulseTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, pulseTime + 0.14);

      osc.connect(gain);
      gain.connect(this.ctx.destination);

      osc.start(pulseTime);
      osc.stop(pulseTime + 0.15);
    }
  }
}

// Global Application State
const soundEngine = new CockpitSoundEngine();
let telemetryChart = null;
let isMonitoring = true;
let sessionSeconds = 0;
let timerInterval = null;

// Initialize Chart.js Waveform
function initTelemetryChart() {
  const canvas = document.getElementById('telemetryChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const maxDataPoints = 30;
  const initialLabels = Array(maxDataPoints).fill('');

  telemetryChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: initialLabels,
      datasets: [
        {
          label: 'EAR Ratio (Proxy)',
          borderColor: '#2563EB',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
          data: Array(maxDataPoints).fill(100)
        },
        {
          label: 'MAR Ratio (Proxy)',
          borderColor: '#D97706',
          backgroundColor: 'rgba(217, 119, 6, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25,
          fill: true,
          data: Array(maxDataPoints).fill(10)
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false }
      },
      scales: {
        x: { display: false },
        y: {
          min: 0.0,
          max: 100,
          grid: { color: 'rgba(55, 65, 81, 0.45)' },
          ticks: {
            color: '#9CA3AF',
            font: { family: 'JetBrains Mono', size: 10 },
            stepSize: 25
          }
        }
      }
    }
  });
}

// Update Chart with New Data Points
function updateChart(visPct, mouthMetric) {
  if (!telemetryChart) return;
  const visDataset = telemetryChart.data.datasets[0].data;
  const mouthDataset = telemetryChart.data.datasets[1].data;

  visDataset.shift();
  visDataset.push(visPct);

  // Scaled mouth metric for visualization (0.0 to 1.0 mapped to 0 to 100)
  mouthDataset.shift();
  mouthDataset.push(mouthMetric * 100.0);

  telemetryChart.update('none');
}

// Update Circular Risk Score Gauge
function updateRiskGauge(score) {
  const circle = document.getElementById('gaugeProgress');
  const scoreVal = document.getElementById('riskScoreVal');
  const label = document.getElementById('riskLevelLabel');
  if (!circle || !scoreVal || !label) return;

  const circumference = 502; // 2 * pi * 80
  const clampedScore = Math.min(100, Math.max(0, score));
  const offset = circumference - (clampedScore / 100) * circumference;
  circle.style.strokeDashoffset = offset;
  scoreVal.textContent = Math.round(clampedScore);

  if (clampedScore < 30) {
    circle.style.stroke = '#16A34A';
    label.className = 'risk-status-pill badge-normal';
    label.textContent = 'ATTENTIVE';
  } else if (clampedScore < 60) {
    circle.style.stroke = '#D97706';
    label.className = 'risk-status-pill badge-warning';
    label.textContent = 'ADVISORY WARNING';
  } else {
    circle.style.stroke = '#DC2626';
    label.className = 'risk-status-pill badge-critical';
    label.textContent = 'CRITICAL ALERT';
  }
}

// Update Navbar Driver Status Pill
function updateStatusPill(statusText, level) {
  const pill = document.getElementById('driverStatusPill');
  const text = document.getElementById('driverStatusText');
  if (!pill || !text) return;

  const display = (statusText || 'ATTENTIVE').toUpperCase();
  text.textContent = `SYSTEM ACTIVE • ${display}`;
  pill.className = 'status-badge';

  if (level === 'CRITICAL') {
    pill.classList.add('badge-critical');
  } else if (level === 'WARNING') {
    pill.classList.add('badge-warning');
  } else {
    pill.classList.add('badge-normal');
  }
}

// Update Telemetry UI with Live Heuristic Pipeline Values
function updateTelemetryUI(data) {
  // 1. Camera Status & Error Handling
  const camBar = document.getElementById('cameraAlertBar');
  const camText = document.getElementById('cameraAlertText');
  if (camBar && camText) {
    if (data.camera_connected === false) {
      camBar.classList.remove('hidden');
      camText.textContent = data.camera_error || 'Camera offline. Check webcam connection or privacy settings.';
    } else {
      camBar.classList.add('hidden');
    }
  }

  // 2. Face Detection Status Badge
  const faceBadge = document.getElementById('faceDetectBadge');
  if (faceBadge) {
    if (data.face_detected) {
      faceBadge.textContent = 'FACE: DETECTED';
      faceBadge.className = 'badge-tag tag-success';
    } else {
      faceBadge.textContent = 'FACE: NOT FOUND';
      faceBadge.className = 'badge-tag tag-danger';
    }
  }

  // 3. Eye Visibility & Blinking
  const visPct = data.visibility_ratio !== undefined ? data.visibility_ratio : 100.0;
  const valVis = document.getElementById('valVisibility');
  const barVis = document.getElementById('barVisibility');
  if (valVis) valVis.textContent = `${Math.round(visPct)}%`;
  if (barVis) {
    barVis.style.width = `${Math.min(100, Math.max(0, visPct))}%`;
    barVis.className = visPct < 40 ? 'track-fill fill-critical' : 'track-fill fill-normal';
  }

  const valEyesCount = document.getElementById('valEyesCount');
  const valEyeAbsence = document.getElementById('valEyeAbsence');
  if (valEyesCount) valEyesCount.textContent = data.eyes_detected || 0;
  if (valEyeAbsence) valEyeAbsence.textContent = `${data.eye_absence_frames || 0} frames`;

  // 4. Mouth Opening Metric (Yawning)
  const mouthMetric = data.mouth_opening_metric || 0.0;
  const valMouth = document.getElementById('valMouth');
  const barMouth = document.getElementById('barMouth');
  if (valMouth) valMouth.textContent = mouthMetric.toFixed(2);
  if (barMouth) {
    barMouth.style.width = `${Math.min(100, Math.max(0, mouthMetric * 250))}%`;
    barMouth.className = data.is_yawning ? 'track-fill fill-warning' : 'track-fill fill-primary';
  }

  // 5. Centroid Head Drift (Approx Distraction)
  const driftX = data.norm_drift_x || 0.0;
  const driftY = data.norm_drift_y || 0.0;
  const valDriftX = document.getElementById('valDriftX');
  const valDriftY = document.getElementById('valDriftY');
  if (valDriftX) valDriftX.textContent = `${driftX >= 0 ? '+' : ''}${driftX.toFixed(2)}`;
  if (valDriftY) valDriftY.textContent = `${driftY >= 0 ? '+' : ''}${driftY.toFixed(2)}`;

  const distractStatus = document.getElementById('distractionStatusText');
  if (distractStatus) {
    if (data.is_distracted) {
      distractStatus.textContent = `${(data.distraction_type || 'DISTRACTED').toUpperCase()} (HEURISTIC)`;
      distractStatus.style.color = 'var(--color-warning)';
    } else {
      distractStatus.textContent = 'ATTENTIVE (HEURISTIC)';
      distractStatus.style.color = 'var(--color-normal)';
    }
  }

  // 6. Bind the 7 Core Telemetry Metric Cards:
  // (EAR, MAR, PERCLOS, FPS, blinks, yawns, alerts)
  const cardValEar = document.getElementById('cardValEar');
  if (cardValEar) {
    cardValEar.textContent = (visPct / 100.0).toFixed(2);
  }

  const cardValMar = document.getElementById('cardValMar');
  if (cardValMar) {
    cardValMar.textContent = mouthMetric.toFixed(2);
  }

  const cardValPerclos = document.getElementById('cardValPerclos');
  if (cardValPerclos) {
    const perclosEstimated = (100 - Math.min(100, Math.max(0, visPct))).toFixed(1);
    cardValPerclos.textContent = `${perclosEstimated}%`;
  }

  const cardValFps = document.getElementById('cardValFps');
  if (cardValFps) {
    cardValFps.textContent = Math.round(data.fps || 30);
  }

  const statBlinks = document.getElementById('statBlinks');
  if (statBlinks) {
    statBlinks.textContent = data.blinks || 0;
  }

  const statYawns = document.getElementById('statYawns');
  if (statYawns) {
    statYawns.textContent = data.yawns || 0;
  }

  const statTotalAlerts = document.getElementById('statTotalAlerts');
  if (statTotalAlerts) {
    const totalCount = (data.drowsiness_alerts || 0) + (data.distraction_alerts || 0);
    statTotalAlerts.textContent = totalCount;
  }

  // 7. FPS Badge in Video Header
  const fpsBadge = document.getElementById('fpsBadge');
  if (fpsBadge) {
    fpsBadge.textContent = `FPS: ${Math.round(data.fps || 30)}`;
  }

  // 8. Circular Risk Gauge & Status Pill
  const risk = data.risk_score || 0.0;
  updateRiskGauge(risk);
  updateStatusPill(data.status || 'ATTENTIVE', data.status_level || 'NORMAL');

  // 9. Alarm Banner & Audio Triggers
  const banner = document.getElementById('videoAlarmBanner');
  const bannerText = document.getElementById('alarmBannerText');

  if (banner && bannerText) {
    if (data.active_alert) {
      banner.classList.remove('hidden');
      bannerText.textContent = data.active_alert.message;

      if (data.active_alert.severity === 'DANGER') {
        soundEngine.playUrgentAlarm();
      } else {
        soundEngine.playChime();
      }
    } else if (data.status_level === 'CRITICAL') {
      banner.classList.remove('hidden');
      bannerText.textContent = `CRITICAL ALERT: ${data.status}`;
      soundEngine.playUrgentAlarm();
    } else {
      banner.classList.add('hidden');
    }
  }

  // 10. Update waveform chart
  updateChart(visPct, mouthMetric);
}

// Fetch Telemetry Snapshot
async function fetchTelemetry() {
  if (!isMonitoring) return;
  try {
    const res = await fetch('/api/telemetry');
    if (res.ok) {
      const data = await res.json();
      updateTelemetryUI(data);
    }
  } catch (err) {
    console.warn('Telemetry fetch error:', err);
  }
}

// Fetch Alerts Log Table
async function fetchAlerts() {
  try {
    const res = await fetch('/api/alerts?limit=15');
    if (res.ok) {
      const data = await res.json();
      renderAlertsTable(data.alerts);
    }
  } catch (err) {
    console.warn('Alerts fetch error:', err);
  }
}

// Render Alerts Table Rows (Pure SVG / Pill Badges, No Emojis)
function renderAlertsTable(alerts) {
  const tbody = document.getElementById('alertsTableBody');
  const countBadge = document.getElementById('alertCountBadge');
  if (!tbody) return;

  if (!alerts || alerts.length === 0) {
    if (countBadge) countBadge.textContent = '0 EVENTS';
    tbody.innerHTML = '<tr class="empty-state-row"><td colspan="4">No safety incidents recorded. Real-time telemetry monitoring active.</td></tr>';
    return;
  }

  if (countBadge) countBadge.textContent = `${alerts.length} EVENTS`;
  tbody.innerHTML = alerts.map(a => {
    let badgeClass = 'badge-tag-sm badge-warning';
    let severityLabel = a.severity || 'WARNING';

    if (a.severity === 'DANGER' || a.alert_type === 'CRITICAL' || (a.risk_score && a.risk_score >= 60)) {
      badgeClass = 'badge-tag-sm badge-critical';
      severityLabel = 'CRITICAL';
    } else if (a.severity === 'ADVISORY' || a.severity === 'WARNING') {
      badgeClass = 'badge-tag-sm badge-warning';
      severityLabel = 'WARNING';
    } else {
      badgeClass = 'badge-tag-sm badge-normal';
      severityLabel = 'NORMAL';
    }

    const typeDisplay = a.alert_type ? `[${a.alert_type}] ` : '';
    const message = a.message || 'Safety event logged.';

    return `
      <tr>
        <td class="mono-num">${a.timestamp || '--:--:--'}</td>
        <td><span class="${badgeClass}">${severityLabel}</span></td>
        <td class="mono-num">${Math.round(a.risk_score || 0)}%</td>
        <td>${typeDisplay}${message}</td>
      </tr>
    `;
  }).join('');
}

// Session Stopwatch Timer
function startTimer() {
  if (timerInterval) clearInterval(timerInterval);
  sessionSeconds = 0;
  timerInterval = setInterval(() => {
    sessionSeconds++;
    const hrs = String(Math.floor(sessionSeconds / 3600)).padStart(2, '0');
    const mins = String(Math.floor((sessionSeconds % 3600) / 60)).padStart(2, '0');
    const secs = String(sessionSeconds % 60).padStart(2, '0');
    const el = document.getElementById('sessionTimer');
    if (el) el.textContent = `${hrs}:${mins}:${secs}`;
  }, 1000);
}

// Toggle Monitoring (Pause / Resume)
async function toggleMonitoring() {
  const btn = document.getElementById('btnStartStop');
  const btnText = document.getElementById('btnStartStopText');

  if (isMonitoring) {
    isMonitoring = false;
    if (btn) btn.className = 'btn-secondary-action';
    if (btnText) btnText.textContent = 'RESUME MONITOR';
    if (timerInterval) clearInterval(timerInterval);
    await fetch('/api/session/stop', { method: 'POST' });
  } else {
    isMonitoring = true;
    if (btn) btn.className = 'btn-secondary-action btn-danger-state';
    if (btnText) btnText.textContent = 'PAUSE MONITOR';
    startTimer();
    const sourceSelect = document.getElementById('selectSource');
    const camIdx = sourceSelect ? parseInt(sourceSelect.value) : 0;
    await fetch('/api/session/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_index: camIdx })
    });
  }
}

// Setup Event Listeners
function setupEventListeners() {
  document.body.addEventListener('click', () => soundEngine.init(), { once: true });

  // Camera Selector
  const selectSource = document.getElementById('selectSource');
  if (selectSource) {
    selectSource.addEventListener('change', async (e) => {
      const camIdx = parseInt(e.target.value);
      await fetch('/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: camIdx })
      });
      const img = document.getElementById('videoFeed');
      if (img) img.src = '/video_feed?' + new Date().getTime();
    });
  }

  // Reconnect button
  const btnReconnect = document.getElementById('btnReconnectCam');
  if (btnReconnect) {
    btnReconnect.addEventListener('click', async () => {
      const sourceSelect = document.getElementById('selectSource');
      const camIdx = sourceSelect ? parseInt(sourceSelect.value) : 0;
      await fetch('/api/camera/reconnect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: camIdx })
      });
      const img = document.getElementById('videoFeed');
      if (img) img.src = '/video_feed?' + new Date().getTime();
    });
  }

  // Start / Stop
  const btnStartStop = document.getElementById('btnStartStop');
  if (btnStartStop) {
    btnStartStop.addEventListener('click', toggleMonitoring);
  }

  // Audio Mute Toggle
  const btnAudio = document.getElementById('btnAudioToggle');
  if (btnAudio) {
    btnAudio.addEventListener('click', () => {
      soundEngine.enabled = !soundEngine.enabled;
      const iconOn = document.getElementById('iconAudioOn');
      const iconOff = document.getElementById('iconAudioOff');
      const label = document.getElementById('audioLabel');

      if (soundEngine.enabled) {
        if (iconOn) iconOn.classList.remove('hidden');
        if (iconOff) iconOff.classList.add('hidden');
        if (label) label.textContent = 'AUDIO ON';
        soundEngine.playChime();
      } else {
        if (iconOn) iconOn.classList.add('hidden');
        if (iconOff) iconOff.classList.remove('hidden');
        if (label) label.textContent = 'AUDIO MUTED';
      }
    });
  }

  // Overlays Toggles
  const toggleFace = document.getElementById('toggleFaceBox');
  if (toggleFace) {
    toggleFace.addEventListener('change', (e) => {
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show_face_box: e.target.checked })
      });
    });
  }

  const toggleEyes = document.getElementById('toggleEyesBox');
  if (toggleEyes) {
    toggleEyes.addEventListener('change', (e) => {
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show_eyes_box: e.target.checked })
      });
    });
  }

  const toggleDrift = document.getElementById('toggleDrift');
  if (toggleDrift) {
    toggleDrift.addEventListener('change', (e) => {
      fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ show_drift_vector: e.target.checked })
      });
    });
  }

  // Settings Modal Handlers
  const modal = document.getElementById('settingsModal');
  const btnOpenModal = document.getElementById('btnSettingsModal');
  const btnCloseModal = document.getElementById('btnCloseModal');

  if (btnOpenModal && modal) {
    btnOpenModal.addEventListener('click', () => modal.classList.remove('hidden'));
  }
  if (btnCloseModal && modal) {
    btnCloseModal.addEventListener('click', () => modal.classList.add('hidden'));
  }

  const sliderEye = document.getElementById('sliderEyeAbsence');
  if (sliderEye) {
    sliderEye.addEventListener('input', (e) => {
      const el = document.getElementById('sliderEyeAbsenceVal');
      if (el) el.textContent = `${e.target.value} frames`;
    });
  }

  const sliderYawn = document.getElementById('sliderYawn');
  if (sliderYawn) {
    sliderYawn.addEventListener('input', (e) => {
      const el = document.getElementById('sliderYawnVal');
      if (el) el.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  const sliderDrift = document.getElementById('sliderDriftX');
  if (sliderDrift) {
    sliderDrift.addEventListener('input', (e) => {
      const el = document.getElementById('sliderDriftXVal');
      if (el) el.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  const sliderVol = document.getElementById('sliderVol');
  if (sliderVol) {
    sliderVol.addEventListener('input', (e) => {
      const el = document.getElementById('sliderVolVal');
      if (el) el.textContent = `${e.target.value}%`;
      soundEngine.volume = e.target.value / 100.0;
    });
  }

  const btnTest = document.getElementById('btnTestChime');
  if (btnTest) {
    btnTest.addEventListener('click', () => soundEngine.playChime());
  }

  const btnSave = document.getElementById('btnSaveSettings');
  if (btnSave && modal) {
    btnSave.addEventListener('click', async () => {
      const eyeFrames = parseInt(document.getElementById('sliderEyeAbsence').value);
      const yawnThresh = parseFloat(document.getElementById('sliderYawn').value);
      const driftX = parseFloat(document.getElementById('sliderDriftX').value);

      await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          eye_absence_consec: eyeFrames,
          yawn_threshold: yawnThresh,
          drift_x_thresh: driftX
        })
      });
      modal.classList.add('hidden');
    });
  }
}

// Bootstrap on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initTelemetryChart();
  setupEventListeners();
  startTimer();

  setInterval(fetchTelemetry, 200);
  setInterval(fetchAlerts, 1000);
});
