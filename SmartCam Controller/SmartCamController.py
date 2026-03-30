import cv2
from ultralytics import YOLO


# Project Name: Smart Phone Trigger
# Description: Camera on , full face show, "Cell Phone" cell phone backpart show cam off.

def start_project():
    # 1. Camera Initialize
    cap = cv2.VideoCapture(0)

    # 2. YOLO Model download (first time e download )
    # yolov8n .It is very fast and can recognize common things
    print("YOLOv8 Model load hochche, somoy lagte pare...")
    model = YOLO("yolov8n.pt")

    # Face detection cascade (for the sake of being simple, using old logic)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("The camera is turning on... If you bring the back part of the phone forward, the camera will turn off.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Camera data pachche na.")
            break

        frame = cv2.flip(frame, 1)  # Mirror

        # --- FACE DETECTION ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- PHONE DETECTION (YOLO) ---
        # conf=0.5 mane hocche model ta jodi 50% When it is confirmed on the phone, it will be detected then .
        results = model(frame, stream=True, conf=0.5)

        for r in results:
            for box in r.boxes:
                # Class name check
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if label == 'cell phone':
                    # Phone detect hoyeche!
                    print("Cell Phone backpart Detected! System shutting down...")

                    # Beep Sound (System tone)
                    print('\a')

                    # Camera immediate release
                    cap.release()
                    cv2.destroyAllWindows()
                    return  # Will exit from the entire program

        # Showing live feed on the screen
        cv2.imshow('Smart Phone Detector', frame)

        # 'q' This tripod will manually close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_project()