import os
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm

def load_classes(classes_file):
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = round(x * dw, 6)
    w = round(w * dw, 6)
    y = round(y * dh, 6)
    h = round(h * dh, 6)
    return (x, y, w, h)

def convert_annotation(input_dir, output_dir, classes):
    annotations_path = os.path.join(input_dir, "annotations")
    labels_output_path = os.path.join(output_dir, "labels")
    os.makedirs(labels_output_path, exist_ok=True)
    
    for xml_file in tqdm(os.listdir(annotations_path)):
        if not xml_file.endswith(".xml"):
            continue
        
        tree = ET.parse(os.path.join(annotations_path, xml_file))
        root = tree.getroot()
        img_size = root.find("size")
        img_w = int(img_size.find("width").text)
        img_h = int(img_size.find("height").text)
        
        label_file = os.path.join(labels_output_path, xml_file.replace(".xml", ".txt"))
        with open(label_file, "w") as out_file:
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in classes:
                    continue  # Ignore classes not in our predefined list
                class_id = classes.index(class_name)
                bbox = obj.find("bndbox")
                b = (float(bbox.find("xmin").text), float(bbox.find("xmax").text),
                     float(bbox.find("ymin").text), float(bbox.find("ymax").text))
                yolo_bbox = convert_bbox((img_w, img_h), b)
                out_file.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format.")
    parser.add_argument("input_dir", type=str, help="Path to the dataset directory containing annotations and images.")
    parser.add_argument("output_dir", type=str, help="Path where YOLO annotations will be saved.")
    parser.add_argument("--classes_file", type=str, default="classes.txt", help="Path to the class names file.")
    args = parser.parse_args()

    classes = load_classes(args.classes_file)
    convert_annotation(args.input_dir, args.output_dir, classes)