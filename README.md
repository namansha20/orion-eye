# 🛰️ ORION-EYE

**Autonomous Asteroid Detection & Evasion System (AADES)**

ORION-EYE is a real-time space debris detection and collision avoidance simulator. It uses a webcam to detect objects (paper balls simulate space debris), tracks their 3D trajectory, assesses collision risk, and recommends evasion maneuvers — all displayed on a live cyberpunk-themed web dashboard.

---

## 📋 Table of Contents

- [What Is This Project?](#-what-is-this-project)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
  - [Mode 1 — Color-Based Standalone (Main.py)](#mode-1--color-based-standalone-mainpy)
  - [Mode 2 — YOLOv8 Desktop (Main2.py)](#mode-2--yolov8-desktop-main2py)
  - [Mode 3 — Web Dashboard (app.py)](#mode-3--web-dashboard-apppy)
- [Training Your Own Model](#-training-your-own-model)
- [Dataset Setup](#-dataset-setup)
- [API Endpoints](#-api-endpoints)
- [System Architecture](#-system-architecture)

---

## 🚀 What Is This Project?

ORION-EYE simulates an onboard AI system for a spacecraft. Using a standard webcam, it:

1. **Detects** spherical objects (paper balls acting as debris proxies) via YOLOv8 or HSV color filtering.
2. **Tracks** their position, velocity, and depth change across frames.
3. **Predicts** where the object will be 15 frames into the future.
4. **Classifies** risk as `CRITICAL`, `HIGH`, or `LOW`.
5. **Recommends** a thrust direction to dodge an incoming collision.

The web interface (`app.py`) provides a full mission-control dashboard with live video, telemetry, event logs, and maneuver planning.

---

## ✨ Features

| Feature | Description |
|---|---|
| Real-Time YOLOv8 Detection | Detects debris at 30+ FPS with confidence & aspect-ratio filtering |
| 3D Trajectory Prediction | Tracks X/Y velocity and Z-axis approach via optical expansion |
| Collision Risk Engine | Three-tier threat classification with color-coded HUD overlays |
| Evasion Maneuver Planner | Calculates optimal thrust direction and Delta-V |
| Web Dashboard | Live MJPEG stream, telemetry API, explainable-AI event log |
| SQLite Event Logging | Persistent, timestamped logs for INFO and CRITICAL events |
| Dual Operation Modes | Standalone OpenCV window **or** Flask web interface |

---

## 📁 Project Structure

```
orion-eye/
├── app.py              # Flask web app — main entry point for web mode
├── Main.py             # Standalone desktop app using HSV color detection
├── Main2.py            # Standalone desktop app using YOLOv8
├── train.py            # YOLOv8 model training script
├── data.py             # Dataset downloader (Roboflow)
├── yolov8n.pt          # Pre-downloaded YOLOv8 nano base weights
├── Find-PaperBalls-1/  # Custom-trained model weights directory
├── templates/
│   └── index.html      # Web dashboard UI
└── PROJECT_DOCUMENTATION.md
```

---

## 🔧 Prerequisites

- Python **3.9 or higher**
- A connected **webcam**
- (Optional) NVIDIA GPU with CUDA for faster inference

---

## 💾 Installation

```bash
# 1. Clone the repository
git clone https://github.com/namansha20/orion-eye.git
cd orion-eye

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install flask opencv-python ultralytics numpy roboflow
```

---

## ▶️ How to Run

### Mode 1 — Color-Based Standalone (`Main.py`)

Uses **HSV color filtering** to track red objects. No trained model required.

```bash
python Main.py
```

- Opens an OpenCV window titled **"AADES Final System"**.
- Point your webcam at a **red spherical object** (e.g., a red ball or crumpled red paper).
- Press **`q`** to quit.

---

### Mode 2 — YOLOv8 Desktop (`Main2.py`)

Uses the **custom-trained YOLOv8 model** for more accurate detection.

1. Update `MODEL_PATH` in `Main2.py` to point to your trained weights:
   ```python
   MODEL_PATH = r"path/to/your/best.pt"
   ```

2. Run:
   ```bash
   python Main2.py
   ```

- Opens an OpenCV window.
- Press **`q`** to quit.

---

### Mode 3 — Web Dashboard (`app.py`)

Full mission-control interface in the browser with live video, telemetry, and logs.

1. Update `MODEL_PATH` in `app.py` to point to your trained weights:
   ```python
   MODEL_PATH = r"path/to/your/best.pt"
   ```

2. Start the server:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

The dashboard shows:
- **Live camera feed** with AR overlays and trajectory trail
- **System Status** panel (objects detected, critical threats)
- **Maneuver Planning** panel (thrust direction, Delta-V)
- **Detected Objects** table (ID, type, estimated distance, risk level)
- **Explainable AI Logs** (real-time event stream from SQLite)

---

## 🧠 Training Your Own Model

The project was trained on a custom paper-ball dataset. To retrain:

1. **Download the dataset** (requires a free Roboflow account):
   ```bash
   python data.py
   ```
   This downloads the `Find-PaperBalls-2` YOLOv8-format dataset.

2. **Run training**:
   ```bash
   python train.py
   ```
   Default settings: **100 epochs**, image size **640**, batch size **8**.  
   Trained weights will be saved to `runs/detect/train*/weights/best.pt`.

3. Update `MODEL_PATH` in `app.py` or `Main2.py` with the path to `best.pt`.

---

## 📦 Dataset Setup

The dataset is sourced from [Roboflow](https://roboflow.com/):

- **Workspace:** `projectsdata`
- **Project:** `find-paperballs`
- **Version:** 2 (YOLOv8 format)

`data.py` handles the download automatically. The resulting folder (`Find-PaperBalls-2/`) contains `data.yaml`, `train/`, `valid/`, and `test/` splits.

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web dashboard |
| `/video_feed` | GET | MJPEG live video stream |
| `/api/telemetry` | GET | JSON: system metrics + last 5 log entries |

**Sample `/api/telemetry` response:**
```json
{
  "metrics": {
    "objects_detected": 1,
    "critical_threats": 0,
    "high_risk": 0,
    "system_status": "TRACKING TARGET",
    "maneuver": "MAINTAIN",
    "delta_v": "0.000",
    "detected_objects": [
      { "id": "OBJ_001", "type": "debris", "distance": "450.00m", "risk": "LOW" }
    ]
  },
  "logs": ["[12:34:56] INFO: Status Change: TRACKING TARGET - Maneuver: MAINTAIN"]
}
```

---

## 🏗️ System Architecture

```
Camera Feed
    │
    ▼
Frame Capture & Flip (OpenCV)
    │
    ▼
AI Detection (YOLOv8) ──── Confidence Filter (≥50%) ──── Aspect Ratio Filter (0.70–1.40)
    │
    ▼
Object Tracking (Deque Buffers — 32 frames)
    │
    ├── Position History (X, Y)
    └── Radius History  (Z proxy)
    │
    ▼
Dynamics Calculation  (dx, dy, growth_rate)
    │
    ▼
Trajectory Prediction  (15-frame lookahead)
    │
    ▼
Collision Risk Assessment
    ├── CRITICAL  →  Approaching + on collision course
    ├── HIGH      →  Trajectory intersect, but receding
    └── LOW       →  Tracked, no imminent threat
    │
    ▼
Evasion Planner  (thrust direction + Delta-V)
    │
    ▼
HUD Overlay / Web Dashboard + SQLite Log
```

---

## 📄 License

This project is released into the **Public Domain**.
