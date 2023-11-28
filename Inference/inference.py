from ultralytics import YOLO
import os

# Load a trained YOLOv8n model
model = YOLO('C:\\Users\\NableIT02\\Desktop\\VDS\\models\\license_plate_detector.pt')

# Directory containing test images
test_image_dir = 'C:\\Users\\NableIT02\\Desktop\\VDS\\Dataset\\test\\images'

# Create a list of image file paths in the test image directory
image_files = [os.path.join(test_image_dir, file) for file in os.listdir(test_image_dir) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Set confidence threshold (adjust as needed)
confidence_threshold = 0.5

# Loop through the list of image file paths and make predictions
for image_file in image_files:
    # Run inference on the current image
    predictions = model.predict(image_file, save=True, imgsz=640, conf=confidence_threshold)

    # You can process or save the predictions as needed
    # For example, you can access the bounding boxes, class labels, and confidence scores in 'predictions' variable
    # You can also save the predicted images using the 'save' parameter in the 'predict' method

