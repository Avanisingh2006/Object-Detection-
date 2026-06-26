import cv2
from ultralytics import YOLO
from gemini_helper import describe_scene

print("Loading YOLO...")
model = YOLO("yolov8n.pt")

print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

print("Camera opened successfully")

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        continue

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, imgsz=416)

    objects = []


    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            objects.append(model.names[cls])

    annotated_frame = results[0].plot()

    frame_count += 1

    if frame_count % 180 == 0:

        unique_objects = list(set(objects))

        if unique_objects:
            description = describe_scene(unique_objects)

            print("\nDetected:", unique_objects)
            print("Gemini:", description)

    cv2.imshow("YOLO + Gemini", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
