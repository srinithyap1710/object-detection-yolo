from ultralytics import YOLO
import cv2
import time

print("STARTING...")

# Load model
model = YOLO("yolov8n.pt")
print("MODEL LOADED")

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

print("CAMERA STARTED")

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not received")
        break

    # Reduce load (optional but good)
    frame = cv2.resize(frame, (320, 240))

    # Run detection
    results = model(frame)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Object Detection", annotated_frame)

    # Small delay (reduces CPU heat)
    time.sleep(0.03)

    # Exit key (ESC or Q)
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
    




