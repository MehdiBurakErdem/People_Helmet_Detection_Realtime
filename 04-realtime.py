import cv2
from ultralytics import YOLO

# Eğittiğimiz modeli yükle
model = YOLO("runs/detect/train-5/weights/best.pt")

# Bilgisayar kamerasını aç
cap = cv2.VideoCapture(0)

while True:

    # Kameradan bir frame al
    ret, frame = cap.read()

    # Frame alınamadıysa devam etme
    if not ret:
        print("Kameradan görüntü alınamadı.")
        break

    # YOLO ile tespit yap
    results = model(frame, conf= 0.55)

    # Tespitleri görüntünün üzerine çiz
    annotated_frame = results[0].plot()

    # Görüntüyü ekranda göster
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # q tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamerayı kapat
cap.release()

# Açılan OpenCV pencerelerini kapat
cv2.destroyAllWindows()