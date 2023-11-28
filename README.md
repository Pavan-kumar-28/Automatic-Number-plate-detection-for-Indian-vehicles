# Automatic-Number-plate-detection-for-Indian-vehicles

VDS(Vehicle detection system)

1)Dataset 
  - Some of the data is Collected from kaggle and some from various websites.
  - The dataset include various images of vehicles like trucks,cars,autorickshaws,motorbikes.
  - Annotated each image through roboflow and converted it into yolov8 format as we are using yolov8 model here.
    https://app.roboflow.com/nable-it/vehicle_license_plate_detection-olpqj/annotate/job/Ds1EN1WDE7VXycMbb9e0
    Roboflow provides tools for annotating and labeling images and videos
  - dataset can be accessed through google drive link which will be attcahed below
  - dataset is divided into Train(around 950 images),val(around 100 images) and test classes

All THE PACKAGES AND MODULES ARE MENTIONED IN REQUIREMENTS.TXT

2)Training
  python version used-- 3.9.18
   - YOLOv8 is the latest version of YOLO by Ultralytics. As a cutting-edge, state-of-the-art (SOTA) model, YOLOv8 builds on the success of previous versions, introducing new features and improvements for enhanced performance, flexibility, and efficiency. YOLOv8 supports a full range of vision AI tasks, including detection, segmentation, pose estimation, tracking, and classification. 
 - the model used here is yolov8 model from ultralytics. can refer more about ultralytics here (https://github.com/ultralytics/ultralytics)

 - Tasks of yolov8- object detection and tracking, instance segmentation, image classification and pose estimation tasks.
   we are using detection task here.
 - there are 5 yolov8 models developed by ultralytics - yolov8n,yolov8s,yolov8m,yolov8l,yolov8
 - yolov8n model is used here out of all models as it takes less time to detect 
 - all packages and modules are mentioned in requirements.txt

DATA.YAML file is used to store the configuration settings for the model, including various hyperparameters and options. This file is used to define how the YOLO model should be trained or used for inference.

 - yolov8m pre trained model is used for training with 100 epochs
 - training accuracy (map) came to be 0.988 after 100 epochs.
 - Training code can be accessed through google drive link.
 - Training Results are saved in vds/training/runs/detect/train.
 - After training, the best model is saved in runs/detect/train/weights. further this model (best.pt) is used for inference

3)Validation

- validation accuracy was seen to be around 0.989

4)Inference 
  - tested the model with 7 videos of truck.
  - inference results are saved in VDS\Inference\runs\detect

MOdel wAS Trained through kaggle for now 
https://www.kaggle.com/code/pavankumar2528/vehicle-license-plate-detection/notebook

google link of entire code named vds is uploaded in mail with this text.


5)How to read the characters inside the license plate

step1) Activating local environment
(base) PS C:\Users\NableIT02> conda activate vds
(vds) PS C:\Users\NableIT02> cd C:\Users\NableIT02\Desktop\vds


step2) install required packages in requirements.txt 
(vds) pip install -r requirements.txt

there are two codes - one for images and one for videos
for images run the code_image.py
for videos run the code_video.py

step3)run the code_images.py to view the images of license_detections along with their text
(vds) python code_image.py

step4)view the results (output) in result folder

there are 4 basic folders in output
 i)detected_images
ii)cropped_images
iii)corrected_images
iv)final_detected_images

step5)run the code_video.py to view the images of license_detections along with their text
(vds) python code_video.py

step6)view the results (output) of videos in result folder
