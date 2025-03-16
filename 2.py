import os
from ultralytics import YOLO

def create_yaml_file():
    """Create the dataset configuration file for YOLOv8."""
    yaml_content = """path: ./datasets/person_detection
train: images/train
val: images/val
test: images/test

names:
  0: person
"""
    with open("data_person.yaml", "w") as f:
        f.write(yaml_content)
    print("Created data_person.yaml")

def train_person_detection():
    """Train the YOLOv8 model for person detection."""
    model = YOLO("yolov8n.pt")
    model.train(data="data_person.yaml", epochs=50, imgsz=640)
    
    os.makedirs("weights", exist_ok=True)

    trained_weights = "runs/detect/train/weights/best.pt"
    if os.path.exists(trained_weights):
        os.rename(trained_weights, "weights/person_detection.pt")
        print("Training complete! Weights saved in weights/person_detection.pt")
    else:
        print("Error: Trained weights not found!")

if __name__ == "__main__":
    create_yaml_file()
    train_person_detection()
