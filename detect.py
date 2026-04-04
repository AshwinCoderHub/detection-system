from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# detect on full test folder
results = model("helmet_dataset/test/images", show=True, save=True)