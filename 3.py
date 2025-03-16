import os
from ultralytics import YOLO

def create_ppe_yaml_file():
    """Create the dataset configuration file for PPE detection."""
    yaml_content = """path: ./datasets/ppe_detection
train: images/train
val: images/val
test: images/test

names:
  0: hard-hat
  1: gloves
  2: mask
  3: glasses
  4: boots
  5: vest
  6: ppe-suit
  7: ear-protector
  8: safety-harness
"""
    with open("data_ppe.yaml", "w") as f:
        f.write(yaml_content)
    print("Created data_ppe.yaml")

def train_ppe_detection():
    """Train the YOLOv8 model for PPE detection."""
    model = YOLO("yolov8n.pt") 
    model.train(data="data_ppe.yaml", epochs=50, imgsz=640)
    
    os.makedirs("weights", exist_ok=True)
    
    trained_weights = "runs/detect/train/weights/best.pt"
    if os.path.exists(trained_weights):
        os.rename(trained_weights, "weights/ppe_detection.pt")
        print("Training complete! Weights saved in weights/ppe_detection.pt")
    else:
        print("Error: Trained weights not found!")

if __name__ == "__main__":
    create_ppe_yaml_file()
    train_ppe_detection()
