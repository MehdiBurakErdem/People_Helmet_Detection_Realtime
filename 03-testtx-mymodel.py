from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model("dataset/test/images/image4.jpg")

result = results[0]

for i, box in enumerate(result.boxes):

    class_id = int(box.cls[0])
    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    class_name = result.names[class_id]

    print(
        f"{i}: "
        f"Class = {class_name}, "
        f"Confidence = {confidence:.2f}, "
        f"Box = ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
    )

results[0].show()