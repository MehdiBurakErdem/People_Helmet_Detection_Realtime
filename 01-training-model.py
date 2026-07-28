from ultralytics import YOLO

# Hazır YOLO11 Nano modelini yüklemiştik, şimdi kırık helmetları ve augment için de total dataset üzerine eski modeli kullanıyroz
model = YOLO("runs/detect/train-3/weights/best.pt")

# Modeli eğit
results = model.train(
    data="DatasetSon/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4
)