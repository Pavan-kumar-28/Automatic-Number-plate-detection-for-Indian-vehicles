from ultralytics import YOLO
import torch
print(torch.cuda.is_available())

# Load a model
model = YOLO("yolov8m.pt")  # load a pretrained model

# Train the model
results = model.train(data='data.yaml', epochs=100, imgsz=640, batch=32)


