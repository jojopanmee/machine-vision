from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import cv2
from analyze import*

# Define paths
PROJECT_ROOT = "Objectdetection"
PROJECT_ROOT_IMAGE = "css-data"
DATA_YAML = os.path.join(PROJECT_ROOT, "dataset.yaml")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create directories if they don't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load a pretrained YOLOv8 model and train it on your dataset
def train_model():
    # Initialize the model
    model = YOLO('yolov8s.pt')  # Load a pretrained small YOLOv8 model
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=200,
        imgsz=640,
        batch=16,
        workers=0,
        patience=15,
        optimizer="AdamW",
        lr0=0.01,
        weight_decay=0.0005,
        augment=True,
        project=RESULTS_DIR,
        name="yolov8s_PPE_detection"
    )
    
    # Save the trained model
    model.save(os.path.join(WEIGHTS_DIR, "best_model.pt"))
    
    return model

# Evaluate the model on the validation set
def evaluate_model(model):
    # Run validation
    val_results = model.val(data=DATA_YAML)
    
    # Print metrics using attributes (no parentheses)
    print(f"Mean Precision: {val_results.box.mp}")
    print(f"Mean Recall: {val_results.box.mr}")
    print(f"mAP@0.5: {val_results.box.map50}")
    print(f"mAP@0.5:0.95: {val_results.box.map}")
    
    return val_results


# Run inference on a sample image
def test_inference(model, image_path):
    # Run inference
    results = model.predict(image_path, conf=0.25)
    
    # Load and process the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Plot results
    plt.figure(figsize=(10, 10))
    plt.imshow(results[0].plot())
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, "inference_example.jpg"))
    plt.close()

# Compare YOLOv5 and YOLOv8 (optional)
def compare_models():
    # You can implement this part to compare different YOLO versions
    pass

if __name__ == "__main__":
    # Train the model
    #model = train_model()
    model = YOLO("Objectdetection/weights/best_model.pt")
    
    # Evaluate the models
    val_results = evaluate_model(model)
    
    # Test on a sample image (replace with your image path)
    test_image = "/home/demo/MachineVisionProject/css-data/test/images/-4405-_png_jpg.rf.82b5c10b2acd1cfaa24259ada8e599fe.jpg"
    test_inference(model, test_image)
