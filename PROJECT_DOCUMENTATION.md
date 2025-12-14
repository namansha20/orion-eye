# ORION-EYE Project Documentation

## 🛰️ Project Overview

**ORION-EYE** is an Autonomous Asteroid Detection and Evasion System (AADES) that simulates real-time space debris tracking and collision avoidance using computer vision and AI. The system combines YOLOv8 object detection with physics-based trajectory prediction to provide intelligent threat assessment and evasion recommendations.

**Project Type:** Computer Vision + AI-based Object Detection and Tracking System  
**Primary Language:** Python  
**Framework:** Flask (Web Interface), OpenCV (Computer Vision), Ultralytics YOLOv8 (AI Detection)  
**License:** Public Domain

---

## 📋 Table of Contents

1. [Features](#-features)
2. [System Architecture](#-system-architecture)
3. [Implementation Map](#-implementation-map)
4. [Project Structure](#-project-structure)
5. [Core Components](#-core-components)
6. [Activities and Workflows](#-activities-and-workflows)
7. [Technical Details](#-technical-details)
8. [Installation and Setup](#-installation-and-setup)
9. [Usage Guide](#-usage-guide)
10. [Dataset Information](#-dataset-information)
11. [API Endpoints](#-api-endpoints)

---

## 🚀 Features

### Core Features

1. **Real-Time Object Detection**
   - YOLOv8-powered detection of paper balls (simulating space debris)
   - Confidence threshold filtering (minimum 50%)
   - Aspect ratio validation (0.70-1.40) for spherical objects
   - Real-time video stream processing at 30+ FPS

2. **Trajectory Prediction**
   - 3D motion tracking (X, Y, Z axes)
   - Velocity vector calculation using deque-based history buffers
   - 15-frame ahead prediction arrow visualization
   - Optical expansion tracking for depth perception (Z-axis)

3. **Collision Detection & Risk Assessment**
   - Dynamic collision zone (80-pixel radius from center)
   - Multi-level threat classification:
     - **CRITICAL**: Approaching object on collision course
     - **HIGH**: Trajectory intersect but receding
     - **LOW**: Target tracked but safe
   - Real-time threat level color coding

4. **Intelligent Evasion Planning**
   - Automated maneuver calculation based on object trajectory
   - Delta-V (velocity change) recommendations
   - Smart thrust direction computation (LEFT/RIGHT, UP/DOWN)
   - Fuel cost estimation

5. **Web-Based Dashboard**
   - Real-time telemetry display
   - Live video feed with AR overlays
   - System status monitoring
   - Event logging with timestamps
   - Responsive cyberpunk-themed UI

6. **Data Logging System**
   - SQLite database for event persistence
   - Timestamped logs with severity levels (INFO, CRITICAL)
   - Historical data retrieval via REST API

### Advanced Features

- **Trail Visualization**: Red trajectory trail showing object's path history (32-frame buffer)
- **HUD Overlay**: Space-themed heads-up display with status indicators
- **Multi-Mode Operation**: Standalone desktop app or web-based interface
- **Explainable AI**: Real-time logging explaining system decisions
- **Edge Case Handling**: Time-critical alerts for objects approaching in <60 seconds

---

## 🏗️ System Architecture

### 10-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Layer 10: User Interface              │
│              (Web Dashboard / Desktop Display)           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Layer 9: API & Communication                │
│         (Flask REST API, WebSocket Streaming)            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           Layer 8: Decision Making & Planning            │
│      (Evasion Calculation, Maneuver Optimization)        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│            Layer 7: Risk Assessment Engine               │
│     (Threat Classification, Collision Probability)       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│          Layer 6: Trajectory Prediction                  │
│    (Physics Modeling, Velocity Calculation, Z-axis)      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│             Layer 5: Object Tracking                     │
│      (Deque-based History, Position/Radius Buffer)       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│            Layer 4: AI Object Detection                  │
│       (YOLOv8 Neural Network, Confidence Filtering)      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         Layer 3: Image Preprocessing                     │
│   (HSV Conversion, Gaussian Blur, Morphology Ops)        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│            Layer 2: Video Acquisition                    │
│        (Camera Interface, Frame Capture, Flip)           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Layer 1: Hardware Interface                 │
│            (Webcam, DirectShow Backend)                  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
Camera Feed → Frame Processing → AI Detection → Object Validation
                                                      ↓
                                               Tracking System
                                                      ↓
                                            Position & Radius History
                                                      ↓
                                          Dynamics Calculation (dx, dy, growth)
                                                      ↓
                                            Trajectory Prediction
                                                      ↓
                                          Collision Risk Assessment
                                                      ↓
                                             Decision Engine
                                           /                  \
                                    Safe Zone            Collision Course
                                          ↓                    ↓
                                  Track & Monitor      Evasion Planning
                                                            ↓
                                                    Maneuver Commands
                                                            ↓
                                                      UI/Dashboard
                                                            ↓
                                                     SQLite Logging
```

---

## 🗺️ Implementation Map

### Module Relationships

```
Main.py (Standalone Desktop)
    ├── OpenCV Video Capture
    ├── Color-based Detection (HSV)
    ├── Trajectory Analysis
    └── Display Window with HUD

Main2.py (YOLOv8 Desktop)
    ├── YOLO Model Loading
    ├── Object Detection & Filtering
    ├── Trajectory Analysis (Shared Logic)
    └── Enhanced Display with AI

app.py (Web Application)
    ├── Flask Server
    ├── VideoCamera Class
    │   ├── YOLO Detection
    │   ├── Trajectory Analysis
    │   └── Frame Streaming
    ├── Database Management (SQLite)
    ├── REST API Endpoints
    │   ├── /video_feed (MJPEG Stream)
    │   └── /api/telemetry (JSON Data)
    └── Template Rendering

train.py (Model Training)
    ├── YOLO Model Initialization
    ├── Dataset Loading (data.yaml)
    └── Training Configuration

data.py (Dataset Downloader)
    └── Roboflow API Integration

templates/index.html (UI)
    ├── Tailwind CSS Styling
    ├── Real-time Dashboard
    ├── JavaScript Telemetry Polling
    └── Object Table & Log Display
```

---

## 📁 Project Structure

```
orion-eye/
├── Main.py                      # Standalone desktop app (color-based detection)
├── Main2.py                     # Standalone desktop app (YOLOv8-based)
├── app.py                       # Flask web application server
├── train.py                     # YOLOv8 model training script
├── data.py                      # Dataset download utility
├── yolov8n.pt                   # Pre-trained YOLOv8 nano weights
├── orion_logs.db                # SQLite database (runtime generated)
├── templates/
│   └── index.html               # Web dashboard UI
├── Find-PaperBalls-1/           # Training dataset
│   ├── data.yaml                # Dataset configuration
│   ├── README.dataset.txt       # Dataset documentation
│   ├── README.roboflow.txt      # Roboflow integration info
│   ├── train/
│   │   ├── images/ (33 images)  # Training images
│   │   └── labels/              # YOLO format labels
│   ├── test/
│   │   ├── images/ (6 images)   # Test images
│   │   └── labels/              # Test labels
│   └── Find-PaperBalls-2/
│       └── valid/
│           ├── images/ (6 images) # Validation images
│           └── labels/            # Validation labels
└── README.md                    # Project overview
```

---

## 🔧 Core Components

### 1. Main.py - Basic Detection System

**Purpose:** Demonstrates color-based tracking without AI

**Key Functions:**
- `calculate_dynamics(pos_history, radius_history)`: Computes X/Y velocity and Z-axis growth rate
- `get_direction_label(dx, dy)`: Translates velocity vectors to human-readable directions
- `main()`: Main event loop with camera capture and visualization

**Detection Method:** HSV color space filtering for red objects
- Red Range 1: H[0-10], S[120-255], V[70-255]
- Red Range 2: H[170-180], S[120-255], V[70-255]

**Output:** Real-time window with trajectory trails and collision warnings

---

### 2. Main2.py - AI-Enhanced Detection

**Purpose:** YOLOv8-based detection with advanced filtering

**Enhancements over Main.py:**
- AI-powered object detection (no color dependency)
- Confidence filtering (50% minimum)
- Aspect ratio validation (0.7-1.4 for spherical objects)
- Camera exposure control for low-light environments

**Configuration:**
```python
CONFIDENCE_MIN = 0.50
RATIO_MIN = 0.70
RATIO_MAX = 1.40
EXPOSURE_VAL = 0
```

---

### 3. app.py - Web Application Server

**Purpose:** Multi-user web interface with persistent logging

**Classes:**

#### `VideoCamera`
- **Lifecycle Management:** Initializes/releases camera resources
- **Frame Processing:** YOLO detection + trajectory analysis
- **State Management:** Maintains position/radius history buffers
- **Encoding:** JPEG compression for MJPEG streaming

**Functions:**

#### `init_db()`
Creates SQLite database with schema:
```sql
CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    level TEXT,
    message TEXT
)
```

#### `log_event(level, message)`
Inserts timestamped events into database

**Global State:**
```python
system_state = {
    "objects_detected": 0,
    "critical_threats": 0,
    "high_risk": 0,
    "system_status": "OK",
    "maneuver": "NONE",
    "delta_v": "0.000",
    "detected_objects": [],
    "last_log": ""
}
```

---

### 4. train.py - Model Training Pipeline

**Purpose:** Fine-tune YOLOv8 on custom paper ball dataset

**Training Parameters:**
- Base Model: `yolov8n.pt` (nano variant, 3.2M parameters)
- Epochs: 100
- Image Size: 640x640
- Batch Size: 8
- Workers: 2

**Output:** Trained weights at `runs/detect/trainX/weights/best.pt`

---

### 5. templates/index.html - Dashboard UI

**Purpose:** Real-time monitoring interface

**Sections:**

1. **Header**: System branding and description
2. **System Status Panel** (Left Column):
   - Object count, critical threats, risk level
   - Current maneuver type and delta-V
   - Edge case alerts

3. **Live Video Feed** (Center):
   - MJPEG stream from `/video_feed`
   - AR overlays with trajectory predictions

4. **Detected Objects Table** (Right Column):
   - Object ID, type, distance, risk level
   - Color-coded risk badges

5. **Explainable AI Logs**:
   - Real-time event stream
   - Decision explanations

**Technologies:**
- Tailwind CSS for styling
- Vanilla JavaScript for API polling (500ms interval)
- Google Fonts: Rajdhani (UI), Share Tech Mono (logs)

---

## 🔄 Activities and Workflows

### Workflow 1: Training a New Model

```
1. Download Dataset
   └─ Run: python data.py
      └─ Roboflow API downloads YOLO-formatted dataset

2. Configure Training
   └─ Edit train.py: Adjust data.yaml path, epochs, batch size

3. Train Model
   └─ Run: python train.py
      └─ YOLOv8 trains for 100 epochs
      └─ Outputs: runs/detect/trainX/weights/best.pt

4. Validate Model
   └─ Check mAP, precision, recall in results
   └─ Test on validation set
```

### Workflow 2: Running Standalone Desktop App

```
1. Basic Color Detection
   └─ Run: python Main.py
      └─ OpenCV window opens
      └─ Place red spherical object in view
      └─ Press 'q' to exit

2. AI-Enhanced Detection
   └─ Update MODEL_PATH in Main2.py
   └─ Run: python Main2.py
      └─ YOLO loads trained model
      └─ Detects paper balls regardless of color
      └─ Press 'q' to exit
```

### Workflow 3: Launching Web Dashboard

```
1. Update Model Path
   └─ Edit app.py: Set MODEL_PATH to your trained weights

2. Start Flask Server
   └─ Run: python app.py
      └─ Server starts on http://0.0.0.0:5000

3. Open Browser
   └─ Navigate to http://localhost:5000
      └─ Dashboard loads with live feed

4. Monitor System
   └─ View telemetry updates every 500ms
   └─ Check logs for decision explanations
   └─ Observe collision warnings and evasion plans
```

### Workflow 4: Decision-Making Process

```
Frame Capture
    ↓
YOLOv8 Detection (confidence > 0.5)
    ↓
Aspect Ratio Filter (0.7 < ratio < 1.4)
    ↓
Update Position & Radius Buffers
    ↓
Calculate Dynamics:
    - dx, dy: Average velocity over 5 frames
    - growth_rate: Radius change (approaching/receding)
    ↓
Predict Future Position (15 frames ahead)
    ↓
Compute Distance to Center
    ↓
Risk Assessment:
    - is_intercept? (distance < 80px)
    - is_approaching? (growth_rate > 0.5)
    ↓
Decision Logic:
    ├─ Intercept + Approaching → CRITICAL: Evasive maneuver
    ├─ Intercept + Receding → HIGH: Monitor closely
    └─ No Intercept → LOW: Track only
    ↓
Execute Actions:
    - Update system_state
    - Log event to database
    - Send telemetry to UI
    - Display HUD warnings
```

---

## 🔬 Technical Details

### Physics Simulation

#### Velocity Calculation (X/Y Plane)
```python
# Average displacement over last 5 frames
dx = Σ(pos[i-1].x - pos[i].x) / 5
dy = Σ(pos[i-1].y - pos[i].y) / 5
```

#### Depth Estimation (Z-axis)
```python
# Optical expansion/contraction
growth_rate = mean(radius[0:5]) - mean(radius[-5:])
# Positive: Approaching, Negative: Receding
```

#### Future Position Prediction
```python
pred_x = current_x + (dx * PREDICTION_FRAMES)
pred_y = current_y + (dy * PREDICTION_FRAMES)
dist_future = ||[pred_x, pred_y] - [center_x, center_y]||
```

### AI Model Details

**Architecture:** YOLOv8n (Nano)
- Parameters: ~3.2 million
- Input: 640×640 RGB images
- Output: Bounding boxes with confidence scores
- Classes: 1 (PaperBall)

**Post-Processing:**
1. Non-Maximum Suppression (NMS)
2. Confidence thresholding (0.5)
3. Aspect ratio filtering (0.7-1.4)
4. Center point extraction: `(x1+x2)/2, (y1+y2)/2`

### Performance Metrics

- **FPS:** 30+ on modern CPUs (with YOLOv8n)
- **Latency:** <33ms per frame
- **Detection Range:** 10-500 pixels radius
- **Prediction Horizon:** 0.5 seconds (15 frames @ 30fps)
- **Collision Zone:** 80-pixel radius (~25% of frame width)

---

## 💻 Installation and Setup

### Prerequisites

```bash
Python >= 3.8
pip >= 21.0
```

### Install Dependencies

```bash
# Computer Vision & AI
pip install opencv-python
pip install numpy
pip install ultralytics  # YOLOv8

# Web Framework
pip install flask

# Dataset Management
pip install roboflow

# Database (built-in)
# sqlite3 comes with Python
```

### Environment Variables (Optional)

```bash
# For data.py
export ROBOFLOW_API_KEY="your_api_key_here"
```

### Hardware Requirements

- **Camera:** USB webcam (0.3MP minimum)
- **CPU:** Intel i5 or equivalent (4+ cores recommended)
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** 500MB for models and dataset

---

## 📖 Usage Guide

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/namansha20/orion-eye.git
cd orion-eye

# 2. Install dependencies
pip install opencv-python numpy ultralytics flask roboflow

# 3. Run standalone app (color-based)
python Main.py

# OR run AI-enhanced app (requires trained model)
python Main2.py

# OR run web dashboard
python app.py
# Open http://localhost:5000 in browser
```

### Training Your Own Model

```bash
# 1. Download dataset
python data.py

# 2. Update data.yaml path in train.py
# Edit line 7: data=r"path/to/your/data.yaml"

# 3. Start training
python train.py

# 4. Wait for 100 epochs (~30-60 minutes on CPU)

# 5. Use trained weights
# Copy runs/detect/trainX/weights/best.pt
# Update MODEL_PATH in Main2.py or app.py
```

### Camera Configuration

```python
# Adjust exposure for low-light environments
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)  # Lower = darker
```

### Tuning Detection Parameters

```python
# Increase for fewer false positives
CONFIDENCE_MIN = 0.60  # Default: 0.50

# Adjust for non-spherical objects
RATIO_MIN = 0.60  # Default: 0.70
RATIO_MAX = 1.50  # Default: 1.40

# Sensitivity
GROWTH_THRESHOLD = 0.3  # Lower = more sensitive to approach
MOVEMENT_THRESHOLD = 3  # Higher = ignore small movements
```

---

## 📊 Dataset Information

### Source
- **Name:** Find PaperBalls
- **Provider:** Roboflow Universe
- **URL:** https://universe.roboflow.com/projectsdata/find-paperballs
- **License:** Public Domain

### Statistics
- **Training Set:** 33 images
- **Test Set:** 6 images
- **Validation Set:** 6 images
- **Total:** 45 images
- **Classes:** 1 (PaperBall)

### Annotation Format
- **Type:** YOLO format (normalized coordinates)
- **Structure:** `class_id x_center y_center width height`
- **Example:** `0 0.512 0.489 0.123 0.145`

### Data Augmentation (Applied during training)
- Random horizontal flips
- Random scaling (0.5x - 1.5x)
- Random translation (±10%)
- HSV color jitter

---

## 🌐 API Endpoints

### GET `/`
**Description:** Serves main dashboard  
**Returns:** HTML page (index.html)  
**Example:**
```bash
curl http://localhost:5000/
```

### GET `/video_feed`
**Description:** MJPEG video stream  
**Content-Type:** `multipart/x-mixed-replace; boundary=frame`  
**Usage:**
```html
<img src="http://localhost:5000/video_feed" />
```

### GET `/api/telemetry`
**Description:** Real-time system telemetry  
**Returns:** JSON  
**Example Response:**
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
      {
        "id": "OBJ_001",
        "type": "debris",
        "distance": "324.56m",
        "risk": "LOW"
      }
    ],
    "last_log": ""
  },
  "logs": [
    "[14:32:15] INFO: Status Change: TRACKING TARGET - Maneuver: MAINTAIN",
    "[14:32:10] INFO: System Boot Complete"
  ]
}
```

**Example Usage:**
```bash
curl http://localhost:5000/api/telemetry
```

**Polling Rate:** 500ms (recommended)

---

## 🎯 Key Algorithms

### 1. Multi-Frame Smoothing
Reduces jitter by averaging velocities:
```python
for i in range(1, 5):
    dx_vals.append(valid_pos[i-1][0] - valid_pos[i][0])
dx = int(np.mean(dx_vals))
```

### 2. Collision Prediction
Extrapolates trajectory to check future intersection:
```python
pred_x = x + (dx * PREDICTION_FRAMES)
pred_y = y + (dy * PREDICTION_FRAMES)
dist_future = np.linalg.norm([pred_x, pred_y] - [center_x, center_y])
is_intercept = dist_future < COLLISION_ZONE
```

### 3. Evasion Calculation
Determines thrust direction opposite to object velocity:
```python
dodge_x = "RIGHT" if dx < 0 else "LEFT"  # Move away from approach
dodge_y = "DOWN" if dy < 0 else "UP"
```

### 4. Trail Rendering
Draws fading trajectory with dynamic thickness:
```python
for i in range(1, len(pos_pts)):
    thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5)
    cv2.line(frame, pos_pts[i-1], pos_pts[i], (0, 0, 255), thickness)
```

---

## 📈 System States

### Status Messages

| State | Meaning | Color | Action |
|-------|---------|-------|--------|
| `SCANNING SECTOR...` | No objects detected | Green | Idle |
| `TRACKING TARGET` | Object detected, safe distance | Yellow | Monitor |
| `TRAJECTORY INTERSECT (SAFE)` | Path crosses but receding | Blue | Monitor |
| `⚠️ COLLISION COURSE` | Approaching on intercept | Red | Evade |

### Maneuver Types

| Type | Description | Delta-V |
|------|-------------|---------|
| `NONE` | No maneuver needed | 0.000 km/s |
| `MAINTAIN` | Maintain current trajectory | 0.000 km/s |
| `THRUST LEFT-UP` | Example evasion command | 1.240 km/s |

---

## 🛡️ Safety and Limitations

### Current Limitations

1. **2D Prediction:** Z-axis depth is estimated via optical expansion (not true 3D)
2. **Single Object:** Tracks one object at a time (highest confidence)
3. **Lighting Dependency:** Color-based mode (Main.py) requires good lighting
4. **Model Specificity:** Trained on paper balls, may not generalize to all objects

### Future Enhancements

- Stereo vision for true depth measurement
- Multi-object tracking with priority queuing
- Reinforcement learning for optimal maneuver planning
- Integration with actual satellite telemetry formats (CCSDS, GMAT)

---

## 🙏 Acknowledgments

- **Dataset:** Roboflow Universe - projectsdata/find-paperballs
- **AI Framework:** Ultralytics YOLOv8
- **Computer Vision:** OpenCV
- **Web Framework:** Flask
- **UI Design:** Tailwind CSS

---

## 📝 License

This project is released under the **Public Domain** license. Feel free to use, modify, and distribute without restrictions.

---

## 📞 Contact & Support

For questions, issues, or contributions:
- **GitHub Repository:** https://github.com/namansha20/orion-eye
- **Report Issues:** Use GitHub Issues tab

---

**Last Updated:** December 14, 2024  
**Version:** 1.0  
**Status:** Production Ready ✅
