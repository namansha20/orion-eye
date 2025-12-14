import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

MODEL_PATH = r"D:\test\Find-PaperBalls-1\runs\detect\train4\weights\best.pt"

BUFFER_SIZE = 32 
PREDICTION_FRAMES = 15 
COLLISION_ZONE = 80 
GROWTH_THRESHOLD = 0.5 
MOVEMENT_THRESHOLD = 2 

CONFIDENCE_MIN = 0.50 
RATIO_MIN = 0.70 
RATIO_MAX = 1.40 

EXPOSURE_VAL = 0 

print(f"🔄 SYSTEM BOOT: Loading AI from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ AI BRAIN ONLINE.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load model.\n{e}")
    exit()

def calculate_dynamics(pos_history, radius_history):
    valid_pos = [p for p in pos_history if p is not None]
    valid_rad = [r for r in radius_history if r is not None]

    if len(valid_pos) < 2 or len(valid_rad) < 5:
        return (0, 0), 0
    
    limit = min(5, len(valid_pos))
    dx_vals = []
    dy_vals = []
    
    for i in range(1, limit):
        dx_vals.append(valid_pos[i-1][0] - valid_pos[i][0])
        dy_vals.append(valid_pos[i-1][1] - valid_pos[i][1])
        
    dx = int(np.mean(dx_vals)) if dx_vals else 0
    dy = int(np.mean(dy_vals)) if dy_vals else 0

    r_now = np.mean(valid_rad[:5])
    r_old = np.mean(valid_rad[-5:]) if len(valid_rad) >= 5 else r_now
    growth_rate = r_now - r_old 
    
    return (dx, dy), growth_rate

def get_direction_label(dx, dy):
    h_dir = ""
    v_dir = ""
    if dx > MOVEMENT_THRESHOLD: h_dir = "RIGHT"
    elif dx < -MOVEMENT_THRESHOLD: h_dir = "LEFT"
    if dy > MOVEMENT_THRESHOLD: v_dir = "DOWN"
    elif dy < -MOVEMENT_THRESHOLD: v_dir = "UP"
    if h_dir == "" and v_dir == "": return "STATIONARY"
    return f"{h_dir} {v_dir}".strip()

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL) 
    
    pos_pts = deque(maxlen=BUFFER_SIZE)
    rad_pts = deque(maxlen=BUFFER_SIZE)

    print("🚀 Autonomous Asteroid Detection & Evasion System")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        
        results = model(frame, stream=True, verbose=False, conf=0.40)
        
        status_msg = "SCANNING SECTOR..."
        status_color = (0, 255, 0) 
        vector_text = "NO TARGET"

        target_found = False
        x, y, radius = 0, 0, 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < CONFIDENCE_MIN: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                obj_w = x2 - x1
                obj_h = y2 - y1
                aspect_ratio = obj_w / float(obj_h)

                if aspect_ratio < RATIO_MIN or aspect_ratio > RATIO_MAX:
                    continue

                target_found = True
                x = x1 + (obj_w // 2)
                y = y1 + (obj_h // 2)
                radius = max(obj_w, obj_h) // 2
                
                pos_pts.appendleft((x, y))
                rad_pts.appendleft(radius)
                break 
            if target_found: break

        if not target_found:
            pos_pts.appendleft(None)
            rad_pts.appendleft(None)

        if target_found:
            (dx, dy), growth_rate = calculate_dynamics(pos_pts, rad_pts)
            direction_label = get_direction_label(dx, dy)
            
            z_label = "APPROACHING" if growth_rate > GROWTH_THRESHOLD else "STABLE"

            pred_x = int(x + (dx * PREDICTION_FRAMES))
            pred_y = int(y + (dy * PREDICTION_FRAMES))
            dist_future = np.linalg.norm(np.array((pred_x, pred_y)) - np.array((center_x, center_y)))
            
            is_intercept = dist_future < COLLISION_ZONE
            is_approaching = growth_rate > GROWTH_THRESHOLD

            if is_intercept and is_approaching:
                status_color = (0, 0, 255) 
                status_msg = "⚠️ COLLISION COURSE"
                
                dodge_x = "RIGHT" if dx < 0 else "LEFT"
                dodge_y = "DOWN" if dy < 0 else "UP"
                cv2.putText(frame, f"ACTION: THRUST {dodge_x} & {dodge_y}", (50, h - 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.line(frame, (int(x), int(y)), (center_x, center_y), (0, 0, 255), 3)

            elif is_intercept and not is_approaching:
                status_color = (255, 100, 0)
                status_msg = "TRAJECTORY INTERSECT (SAFE)"
            else:
                status_color = (0, 255, 255)
                status_msg = "TRACKING TARGET"

            vector_text = f"V: {direction_label} | Z: {z_label}"
            
            cv2.circle(frame, (int(x), int(y)), int(radius), status_color, 2)
            cv2.circle(frame, (int(x), int(y)), 2, status_color, -1)
            
            if abs(dx) > 1 or abs(dy) > 1:
                cv2.arrowedLine(frame, (int(x), int(y)), (pred_x, pred_y), (0, 255, 255), 3)

        for i in range(1, len(pos_pts)):
            if pos_pts[i - 1] is None or pos_pts[i] is None: continue
            
            thickness = int(np.sqrt(BUFFER_SIZE / float(i + 1)) * 2.5)
            cv2.line(frame, pos_pts[i - 1], pos_pts[i], (0, 0, 255), thickness)

        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.putText(frame, "AADES AUTONOMOUS SENSOR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(frame, status_msg, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        cv2.putText(frame, vector_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (100, 100, 100), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (100, 100, 100), 1)
        cv2.circle(frame, (center_x, center_y), COLLISION_ZONE, (50, 50, 50), 1)

        cv2.imshow("AADES Final System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()