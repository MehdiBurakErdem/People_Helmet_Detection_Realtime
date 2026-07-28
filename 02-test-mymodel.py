from ultralytics import YOLO

model = YOLO("runs/detect/train-5/weights/best.pt")

#Confidence değeri 0.25'in altında olan detection'ları gösterme
results1 = model("datasetSon/test/images/suggested-GYjETikViqwklkqvjUMH_jpg.rf.7b8d2d7e03c299ab3fa94a01500c780f.jpg") 
results2 = model("datasetSon/test/images/image3.jpg")

results1[0].show()
results2[0].show()