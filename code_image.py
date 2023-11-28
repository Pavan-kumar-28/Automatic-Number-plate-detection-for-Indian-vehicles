import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
# from util import read_license_plate
import os

# Path to YOLOv8 model for license plate detection
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'

# Create a YOLO model for license plate detection
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# Initialize EasyOCR reader for text recognition
reader = easyocr.Reader(['en'], gpu=False)

# Function to detect and print license plates
def detect_license_plates(image_path):
    img = cv2.imread(image_path)
    img_to_analyze = img.copy()
    img_to_analyze = cv2.cvtColor(img_to_analyze, cv2.COLOR_BGR2RGB)

    # Assume license_plate_detector is defined globally
    license_detections = license_plate_detector(img_to_analyze)[0]

    if len(license_detections.boxes.cls.tolist()) != 0:
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_detected = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            return license_plate_detected


def cropped_license_plates(image_path):
    img = cv2.imread(image_path)
    img_to_analyze = img.copy()
    img_to_analyze = cv2.cvtColor(img_to_analyze, cv2.COLOR_BGR2RGB)

    # Assume license_plate_detector is defined globally
    license_detections = license_plate_detector(img_to_analyze)[0]

    if len(license_detections.boxes.cls.tolist()) != 0:
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]

            return license_plate_crop


def calculate_tilt_angle(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1)
        angles.append(angle)

    tilt_angle = np.median(angles)
    return tilt_angle


def rotate_image(image, angle):
    rotated_image = cv2.warpAffine(image, cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle * 180 / np.pi, 1),
                                   (image.shape[1], image.shape[0]))
    return rotated_image


# def read_license_plate(corrected_image, img1):
#     scores = 0
#     detections = reader.readtext(corrected_image)

#     if not detections:
#         return None, None

#     rectangle_size = corrected_image.shape[0] * corrected_image.shape[1]
#     plate = []

#     for result in detections:
#         length = np.sum(np.subtract(result[0][1], result[0][0]))
#         height = np.sum(np.subtract(result[0][2], result[0][1]))

#         if length * height / rectangle_size > 0.17:
#             bbox, text, score = result
#             text = text.upper()
#             scores += score
#             plate.append(text)

#     if plate:
#         return " ".join(plate), scores / len(plate)
#     else:
#         return " ".join(plate), 0
    

def read_license_plate(corrected_image):
    detections = reader.readtext(corrected_image)

    if not detections:
        return None, None

    plate = []
    total_score = 0

    for bbox, text, score in detections:
        length = np.sum(np.subtract(bbox[1], bbox[0]))
        height = np.sum(np.subtract(bbox[2], bbox[1]))

        rectangle_size = corrected_image.shape[0] * corrected_image.shape[1]
        if length * height / rectangle_size > 0.17:
            text = text.upper()
            plate.append(text)
            total_score += score

    if plate:
        return " ".join(plate), total_score / len(plate)
    else:
        return None, None


def main():
    input_image_path = './test'
    output_path = './results'
    detected_folder = 'detected_images'
    cropped_folder = 'cropped_images'
    corrected_folder = 'corrected_images'
    final_output = 'final_detected_images'

    # Create the output directory and subfolders if they don't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, detected_folder), exist_ok=True)
    os.makedirs(os.path.join(output_path, cropped_folder), exist_ok=True)
    os.makedirs(os.path.join(output_path, corrected_folder), exist_ok=True)
    os.makedirs(os.path.join(output_path, final_output), exist_ok=True)

    image_files = [os.path.join(input_image_path, file) for file in os.listdir(input_image_path) if
                   file.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for idx, image_file in enumerate(image_files):
        img1 = cv2.imread(image_file)
        img_to_analyze = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

        license_plate_detected_img = detect_license_plates(image_file)
        # Check if a license plate is detected
        if license_plate_detected_img is not None:
            # Save the detcted license plate image
            detected_output_file = os.path.join(output_path, detected_folder, f'detected_{idx}.jpg')
            cv2.imwrite(detected_output_file, license_plate_detected_img)

        license_plate_crop_img = cropped_license_plates(image_file)
        if license_plate_crop_img is not None:
            # Save the cropped license plate image
            cropped_output_file = os.path.join(output_path, cropped_folder, f'cropped_{idx}.jpg')
            cv2.imwrite(cropped_output_file, license_plate_crop_img)

            # Perform other operations (Hough lines, tilt correction, etc.)
            gray = cv2.cvtColor(license_plate_crop_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 75, maxLineGap=35)

            # Check if lines is not None 
            if lines is not None:
                img = license_plate_crop_img.copy()
                tilt_angle = calculate_tilt_angle(lines)
                corrected_image = rotate_image(img, tilt_angle)

                # Save the corrected image
                corrected_output_file = os.path.join(output_path, corrected_folder, f'corrected_{idx}.jpg')
                cv2.imwrite(corrected_output_file, corrected_image)

                # Assume license_plate_detector is defined globally
                license_detections = license_plate_detector(img_to_analyze)[0]

                if len(license_detections.boxes.cls.tolist()) != 0:
                    for license_plate in license_detections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate
                        cv2.rectangle(img1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                        license_plate_text, license_plate_text_score = read_license_plate(corrected_image)
                        cv2.rectangle(img1, (int(x1) - 40, int(y1) - 40), (int(x2) + 70, int(y1)), (255, 255, 255), cv2.FILLED)
                        cv2.putText(img1,
                            str(license_plate_text),
                            (int((int(x1) + int(x2)) / 2) - 60, int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 0), 
                            3)
                        final_output_file = os.path.join(output_path, final_output, f'final_detcted_{idx}.jpg')
                        cv2.imwrite(final_output_file, img1)
                # cv2.imwrite('./results/detected',img)  
                # cv2.imshow('pic',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()