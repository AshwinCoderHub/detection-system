from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="helmet_dataset/data.yaml",
    epochs=50,
    imgsz=320,
    batch=4,
    workers=2,
    device=0
)