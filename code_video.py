import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import os

# Path to YOLOv8 model for license plate detection
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'

# Create a YOLO model for license plate detection
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# Initialize EasyOCR reader for text recognition
reader = easyocr.Reader(['en'], gpu=False)

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


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        img_to_analyze = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Assume license_plate_detector is defined globally
        license_detections = license_plate_detector(img_to_analyze)[0]
        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # license_plate_detected = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                # cv2.imshow("Processed Video", license_plate_detected)
                # # Write the frame to the output video
                # out.write(frame)
                # # Break the loop if 'q' key is pressed
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Perform other operations (Hough lines, tilt correction, etc.)
                gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 75, maxLineGap=35)

                # Check if lines is not None 
                if lines is not None:
                    img = license_plate_crop.copy()
                    tilt_angle = calculate_tilt_angle(lines)
                    corrected_image = rotate_image(img, tilt_angle)

                    # Assume license_plate_detector is defined globally
                    license_detections = license_plate_detector(img_to_analyze)[0]

                    if len(license_detections.boxes.cls.tolist()) != 0:
                        for license_plate in license_detections.boxes.data.tolist():
                            x1, y1, x2, y2, _, _ = license_plate
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                            license_plate_text, _ = read_license_plate(corrected_image)
                            cv2.rectangle(frame, (int(x1) - 40, int(y1) - 40), (int(x2) + 70, int(y1)), (255, 255, 255), cv2.FILLED)
                            cv2.putText(frame,
                                        str(license_plate_text),
                                        (int((int(x1) + int(x2)) / 2) - 60, int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 0, 0), 
                                        3)

                                # Display the frame
                            cv2.imshow("Processed Video", frame)

                                # Write the frame to the output video
                            out.write(frame)

                                # Break the loop if 'q' key is pressed
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = './test/truckentry1.mp4'
    output_video_path = './results.avi'
    process_video(input_video_path, output_video_path)


        
        
        # processed_frame = process_frame(frame)

        #     # Display the frame
        # cv2.imshow("Processed Video", processed_frame)

        #     # Write the frame to the output video
        # out.write(processed_frame)

        #         # Break the loop if 'q' key is pressed
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     input_video_path = './test/sample.mp4'
#     output_video_path = './results.mp4'
#     process_video(input_video_path, output_video_path)

