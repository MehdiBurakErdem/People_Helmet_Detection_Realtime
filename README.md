# YOLO Human & Helmet (Hard Hat) Detection
An object detection system built with a custom-trained YOLO11n model to detect people and safety helmets in images and real-time video streams.
The project covers the complete computer vision workflow, including dataset creation, annotation, model training, evaluation, and iterative improvement.


## Overview
The goal of this project is to build a real-time safety helmet detection system capable of:
* Detecting people in images and camera streams
* Detecting safety helmets worn by people
* Evaluating model performance using object detection metrics
* Running real-time inference using a webcam
* Exploring the detection of broken or damaged helmets

The current model achieves approximately 0.87 mAP50 on the test dataset and can perform real-time detection using a webcam.


## Technologies
* Python
* YOLO11n
* Ultralytics
* OpenCV
* Roboflow
* PyTorch


## Dataset
The dataset was created using a combination of manual and semi-automatic annotation workflows in Roboflow.
The current detection model contains two classes:
* person
* helmet

The annotations follow the standard YOLO format:
class_id x_center y_center width height
All bounding box coordinates are normalized between 0 and 1.
The dataset is divided into three subsets:
* train
* valid
* test

During the development process, the dataset evolved from approximately 20 manually labeled images to around 266 raw images. After applying data augmentation techniques, the dataset contained approximately 750 training samples.
The augmentation process included:
* Rotation
* Horizontal flipping
* Cropping
* Brightness adjustments
* Contrast adjustments

The dataset was also improved to include a more balanced representation of:
* People wearing helmets
* People without helmets
* Damaged or broken helmet examples


## Model Training
The project uses the YOLO11n model from the Ultralytics framework.
The model was trained on a custom dataset containing two object classes:
0 → person
1 → helmet

The training and evaluation process included monitoring metrics such as:
* Precision
* Recall
* mAP50
* mAP50-95

The current model achieves approximately:
mAP50 ≈ 0.87
The trained model can also perform real-time object detection using a webcam.


## Current Limitations
Broken or damaged helmet detection is not yet reliable.
The current object detection model only contains two classes:
* person
* helmet

Although damaged helmet examples are present in the dataset, there are currently not enough diverse and representative samples to train a robust system capable of reliably distinguishing between intact and damaged helmets.
This remains one of the main areas for future development.


## Roadmap
Future improvements planned for the project include:
* Collecting more broken and damaged helmet images
* Increasing the diversity of the dataset
* Balancing the number of intact helmet, broken helmet, and no-helmet examples
* Evaluating a separate second-stage classification model for helmet condition
* Comparing YOLO11n with larger model variants such as YOLO11s
* Evaluating the trade-off between detection accuracy and inference speed
* Testing the system under real-world conditions
* Evaluating performance under different lighting conditions
* Testing different camera distances and viewing angles


## Future Architecture
The planned system may use a two-stage approach:
Input Image / Video
↓
Person & Helmet Detection
↓
Helmet Detection
↓
Helmet Condition Classification
↓
Intact / Damaged

This approach will be evaluated once a sufficiently large and diverse dataset of damaged helmets has been collected.


## Contributing
If you have additional images of broken or damaged safety helmets, or if you can help with dataset annotation, contributions are welcome.
The main bottleneck of the current project is the lack of a sufficiently large and diverse dataset for reliable damaged-helmet detection.
Any contribution that helps expand and improve this part of the dataset is greatly appreciated.
