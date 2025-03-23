#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc

# Define paths
PROJECT_ROOT = "Objectdetection"
DATA_YAML = os.path.join(PROJECT_ROOT, "dataset.yaml")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Create directories if they don't exist
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pretrained weight file for YOLOv5 (ensure this file exists)
WEIGHT_FILE = 'yolov5n.pt'

def train_model():
    if not os.path.exists(WEIGHT_FILE):
        raise FileNotFoundError(f"Weight file {WEIGHT_FILE} not found. Please download it or adjust the path.")
    print(f"Loading weights from: {WEIGHT_FILE}")
    model = YOLO(WEIGHT_FILE)
    results = model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=16,
        workers=0,
        patience=15,
        optimizer="AdamW",
        lr0=0.01,
        weight_decay=0.0005,
        augment=True,
        project=RESULTS_DIR,
        name="yolov5_PPE_detection"
    )
    model.save(os.path.join(WEIGHTS_DIR, "best_yolov5.pt"))
    return model

def evaluate_model(model):
    val_results = model.val(data=DATA_YAML)
    print(f"Mean Precision: {val_results.box.mp}")
    print(f"Mean Recall: {val_results.box.mr}")
    print(f"mAP@0.5: {val_results.box.map50}")
    print(f"mAP@0.5:0.95: {val_results.box.map}")
    return val_results

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - YOLOv5')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve_yolov5.png"))
    plt.close()

def plot_overfitting_graph(training_errors, validation_errors):
    epochs = range(1, len(training_errors) + 1)
    plt.figure()
    plt.plot(epochs, training_errors, label='Training Error', color='blue')
    plt.plot(epochs, validation_errors, label='Validation Error', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Overfitting vs Underfitting - YOLOv5')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "overfitting_yolov5.png"))
    plt.close()

def test_inference(model, image_path):
    results = model.predict(image_path, conf=0.25)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(results[0].plot())
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, "inference_yolov5.jpg"))
    plt.close()

if __name__ == "__main__":
    model = train_model()
    evaluate_model(model)
    
    # For demonstration, we use simulated ROC data.
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)
    plot_roc_curve(y_true, y_scores)
    
    # Simulated training and validation error curves (replace with your actual logs)
    training_errors = np.random.rand(150) * 0.1 + np.linspace(0.5, 0.1, 150)
    validation_errors = np.random.rand(150) * 0.1 + np.linspace(0.6, 0.3, 150)
    plot_overfitting_graph(training_errors, validation_errors)
    
    # Inference on a sample image (update the path to your test image)
    test_image = "/home/demo/MachineVisionProject/css-data/test/images/sample.jpg"
    test_inference(model, test_image)
