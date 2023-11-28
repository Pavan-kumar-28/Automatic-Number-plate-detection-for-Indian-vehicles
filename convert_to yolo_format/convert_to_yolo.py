import os
import xml.etree.ElementTree as ET

# Define the class label and its corresponding ID
class_label = "license_plate"
class_id = 0  # You can adjust this if you have more classes

def convert_xml_to_yolo(xml_file, yolo_file):
    with open(xml_file, 'r') as xml:
        tree = ET.parse(xml)
        root = tree.getroot()

        with open(yolo_file, 'w') as yolo:
            for obj in root.findall('.//object'):
                # Use the assigned class_id as the label
                label = str(class_id)

                bndbox = obj.find('bndbox')

                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                width = float(root.find('.//size/width').text)
                height = float(root.find('.//size/height').text)

                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                yolo_line = f"{label} {x_center} {y_center} {box_width} {box_height}\n"
                yolo.write(yolo_line)

def create_classes_file(output_folder):
    # Create a "classes.txt" file in the output folder
    classes_file = os.path.join(output_folder, "classes.txt")
    with open(classes_file, 'w') as classes:
        classes.write(class_label)

def batch_convert_annotations(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".xml"):
            xml_file = os.path.join(input_folder, filename)
            yolo_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")
            convert_xml_to_yolo(xml_file, yolo_file)

    # Create the "classes.txt" file
    create_classes_file(output_folder)

if __name__ == "__main__":
    input_folder = "C:\\Users\\NableIT02\\Desktop\\VDS\\Dataset\\test\\labels"  # Replace with the path to your folder of XML files
    output_folder = "C:\\Users\\NableIT02\\Desktop\\VDS\\Dataset\\test\\labels_new"  # Replace with the desired output directory
    batch_convert_annotations(input_folder, output_folder)