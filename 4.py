import os
import cv2
import argparse
from ultralytics import YOLO

def load_model(model_path):
    """Load a YOLOv8 model."""
    return YOLO(model_path)

def run_inference(input_dir, output_dir, person_model_path, ppe_model_path):
    """Run inference for person and PPE detection models and save results."""
    person_model = load_model(person_model_path)
    ppe_model = load_model(ppe_model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        person_results = person_model(img_path)
        cropped_persons = []
        
        for result in person_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = image[y1:y2, x1:x2]
                cropped_persons.append((cropped_img, (x1, y1, x2, y2)))
        
        for idx, (cropped_img, (x1, y1, x2, y2)) in enumerate(cropped_persons):
            if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
                ppe_results = ppe_model(cropped_img)
                
                for ppe_result in ppe_results:
                    for box in ppe_result.boxes:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        label = ppe_model.names[class_id]
                        
                        fx1 = x1 + px1
                        fy1 = y1 + py1
                        fx2 = x1 + px2
                        fy2 = y1 + py2
                        
                        cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                        cv2.putText(image, label, (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        output_path = os.path.join(output_dir, img_file)
        cv2.imwrite(output_path, image)
        print(f"Processed {img_file} and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing input images")
    parser.add_argument("output_dir", type=str, help="Directory to save output images")
    parser.add_argument("person_det_model", type=str, help="Path to trained person detection model")
    parser.add_argument("ppe_detection_model", type=str, help="Path to trained PPE detection model")
    args = parser.parse_args()
    
    run_inference(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)
