import easyocr
import base64
from PIL import Image
import numpy as np
# from code_rotation import rotate_image

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
                                                'license_plate_bbox_score', 'license_number', 'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and 'license_plate' in results[frame_nmr][car_id].keys() and 'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score']))
        f.close()

# def read_license_plate(license_plate_crop):
#     """
#     Read the license plate text from the given cropped image.

#     Args:
#         license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

#     Returns:
#         tuple: Tuple containing the formatted license plate text and its confidence score.
#     """

#     detections = reader.readtext(license_plate_crop)
#     print(detections)

#     if detections == []:
#         return None, None

#     for detection in detections:
#         bbox, text, score = detection

#         text = text.upper()
#         print(text)

#         if text is not None and score is not None and bbox is not None and len(text) >= 6:
#             return text, score

#     return None, None

def read_license_plate(corrected_image, img1):
    scores = 0
    detections = reader.readtext(corrected_image)

    if not detections:
        return None, None

    rectangle_size = corrected_image.shape[0] * corrected_image.shape[1]
    plate = []

    for result in detections:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > 0.17:
            bbox, text, score = result
            text = text.upper()
            scores += score
            plate.append(text)

    if plate:
        return " ".join(plate), scores / len(plate)
    else:
        return " ".join(plate), 0