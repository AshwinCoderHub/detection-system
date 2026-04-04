from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="helmet_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)