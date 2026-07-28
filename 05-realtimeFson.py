from ultralytics import YOLO
import cv2

# =========================================================
# 1. MODELLERİ YÜKLE
# =========================================================

# Model 1: İnsan + Helmet tespiti
model1 = YOLO("runs/detect/train-5/weights/best.pt")

# Model 2: Crack tespiti
model2 = YOLO("runs/detect/train-4/weights/best.pt")


# =========================================================
# 2. GÖRÜNTÜYÜ OKU
# =========================================================

image_path = "datasetSon/test/images/image2.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Görüntü bulunamadı!")
    exit()


# =========================================================
# 3. MODEL 1 → İNSAN VE HELMET TESPİTİ
# =========================================================

results1 = model1.predict(
    source=image,
    conf=0.5
)


# Model 1'in class isimleri
names1 = model1.names


# =========================================================
# 4. MODEL 1 SONUÇLARINI GEZ
# =========================================================

for result in results1:

    boxes = result.boxes

    for box in boxes:

        # Class ID
        class_id = int(box.cls[0])

        # Confidence
        confidence = float(box.conf[0])

        # Bounding Box koordinatları
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_name = names1[class_id]

        print(
            f"Model 1 -> {class_name} "
            f"Confidence: {confidence:.2f}"
        )


        # =================================================
        # 5. SADECE HELMET BULUNDUĞUNDA MODEL 2 ÇALIŞSIN
        # =================================================

        if class_name.lower() == "helmet":

            # Helmet bounding box'ını çiz
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"Helmet {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            # =================================================
            # 6. HELMET'İ CROP ET
            # =================================================

            helmet_crop = image[y1:y2, x1:x2]

            if helmet_crop.size == 0:
                continue


            # =================================================
            # 7. MODEL 2 → CRACK TESPİTİ
            # =================================================

            results2 = model2.predict(
                source=helmet_crop,
                conf=0.3
            )


            # =================================================
            # 8. CRACK SONUÇLARINI AL
            # =================================================

            for result2 in results2:

                boxes2 = result2.boxes

                for box2 in boxes2:

                    crack_confidence = float(box2.conf[0])

                    # Model 2 crop koordinatları
                    cx1, cy1, cx2, cy2 = map(
                        int,
                        box2.xyxy[0]
                    )


                    # =========================================
                    # 9. CROP KOORDİNATLARINI
                    #    ORİJİNAL GÖRÜNTÜYE GERİ TAŞI
                    # =========================================

                    original_x1 = x1 + cx1
                    original_y1 = y1 + cy1

                    original_x2 = x1 + cx2
                    original_y2 = y1 + cy2


                    # =========================================
                    # 10. CRACK BOUNDING BOX ÇİZ
                    # =========================================

                    cv2.rectangle(
                        image,
                        (original_x1, original_y1),
                        (original_x2, original_y2),
                        (0, 0, 255),
                        3
                    )


                    cv2.putText(
                        image,
                        f"CRACK {crack_confidence:.2f}",
                        (
                            original_x1,
                            original_y1 - 10
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                    print(
                        f"CRACK bulundu! "
                        f"Confidence: "
                        f"{crack_confidence:.2f}"
                    )


# =========================================================
# 11. SONUCU GÖSTER
# =========================================================

cv2.imshow(
    "Helmet + Crack Detection",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()