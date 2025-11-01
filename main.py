import cv2
import time
from ultralytics import YOLO
import numpy as np

# -------------------------------
# 1️⃣ Load YOLOv8 model
# -------------------------------
model = YOLO("yolov8n.pt")  # pre-trained small model
objects_of_interest = ["cell phone", "book"]  # objects to monitor

# -------------------------------
# 2️⃣ Load Haar cascades for face & eyes
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# -------------------------------
# 3️⃣ Initialize webcam
# -------------------------------
cap = cv2.VideoCapture(0)

# -------------------------------
# 4️⃣ Eye contact tracking variables
# -------------------------------
no_eye_contact_start = None
violation_count = 0
VIOLATION_THRESHOLD = 5  # seconds

# -------------------------------
# 5️⃣ Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # 5a️⃣ Detect faces and eyes
    # -------------------------------
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_detected = False

    # Draw faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eyes_detected = True

    # -------------------------------
    # 5b️⃣ Eye contact violation
    # -------------------------------
    current_time = time.time()
    if not eyes_detected:
        if no_eye_contact_start is None:
            no_eye_contact_start = current_time
        elif current_time - no_eye_contact_start > VIOLATION_THRESHOLD:
            violation_count += 1
            print(f"[VIOLATION] No eye contact for {VIOLATION_THRESHOLD}s (Total: {violation_count})")
            no_eye_contact_start = current_time
    else:
        no_eye_contact_start = None

    # -------------------------------
    # 5c️⃣ Check for multiple persons
    # -------------------------------
    if len(faces) > 1:
        print(f"[WARNING] Multiple people detected! ({len(faces)} faces on screen)")
        cv2.putText(frame, "WARNING: Multiple People Detected!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # -------------------------------
    # 5d️⃣ Prepare frame for object detection (ignore face regions)
    # -------------------------------
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    for (x, y, w, h) in faces:
        mask[y:y + h, x:x + w] = 0  # ignore face region
    frame_for_detection = cv2.bitwise_and(frame, frame, mask=mask)

    # -------------------------------
    # 5e️⃣ Object detection
    # -------------------------------
    results = model.predict(frame_for_detection, verbose=False)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label in objects_of_interest and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w_box, h_box = x2 - x1, y2 - y1

                # Ignore very small boxes
                if w_box * h_box < 2000:
                    continue

                # Ignore objects overlapping faces
                overlap = False
                for (fx, fy, fw, fh) in faces:
                    if x1 < fx + fw and x2 > fx and y1 < fy + fh and y2 > fy:
                        overlap = True
                        break
                if overlap:
                    continue

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print(f"[VIOLATION] {label} detected (Total objects: 1)")

    # -------------------------------
    # 6️⃣ Display frame
    # -------------------------------
    cv2.imshow("Interview Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# 7️⃣ Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()