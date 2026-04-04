import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# ---------------- MENU ----------------
print("\n===== Helmet Detection System =====")
print("1 - Offline Video")
print("2 - Laptop Camera")
print("3 - RTSP Stream")

choice = input("Enter your choice (1/2/3): ")

# ---------------- SOURCE SELECTION ----------------
if choice == "1":
    source = "/home/ashwin/Desktop/python _fundamentals/WhatsApp Video 2026-04-03 at 3.52.08 PM.mp4"
elif choice == "2":
    source = 0
elif choice == "3":
    source = input("Enter RTSP URL: ")
else:
    print("❌ Invalid choice")
    exit()

# ---------------- VIDEO CAPTURE ----------------
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("❌ Error opening video source")
    exit()

# ---------------- OPTIONAL: SAVE OUTPUT ----------------
save_output = input("Do you want to save output video? (y/n): ")

if save_output.lower() == 'y':
    width = int(cap.get(3))
    height = int(cap.get(4))
    out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        20,
        (width, height)
    )
else:
    out = None

# ---------------- PROCESS LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = model.names[cls]

        # Custom labels
        if label == "helmet":
            text = f"HELMET ✅ {conf:.2f}"
            color = (0, 255, 0)
        else:
            text = f"NO HELMET ❌ {conf:.2f}"
            color = (0, 0, 255)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Put text
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Title overlay
    cv2.putText(frame, "Helmet Detection System",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    # Show frame
    cv2.imshow("Helmet Detection Demo", frame)

    # Save output
    if out is not None:
        out.write(frame)

    # Exit key
    if cv2.waitKey(1) == 27:
        break

# ---------------- CLEANUP ----------------
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

print("✅ Program finished successfully")