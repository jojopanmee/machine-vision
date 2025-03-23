import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob

# ==============================
# CNN Safety Classifier
# ==============================
class PPESafetyClassifier(nn.Module):
    def __init__(self, input_size=128):
        super(PPESafetyClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        conv_output_size = input_size // 16  # 4 pooling layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * conv_output_size * conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2 classes: safe (1) or unsafe (0)
        )
        # Image transformations (matching your CNN input size)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def preprocess_image(self, image):
        """Convert a NumPy image (or PIL image) to a tensor."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0)  # add batch dimension

    def predict_safety(self, image, device='cpu'):
        """Predict safety (0=unsafe, 1=safe) for a given person ROI."""
        self.eval()
        self.to(device)
        with torch.no_grad():
            image_tensor = self.preprocess_image(image).to(device)
            output = self(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        return prediction, confidence

# ==============================
# Dataset for CNN training (expects subdirectories: safe/ and unsafe/)
# ==============================
class SafetyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        safe_dir = os.path.join(image_dir, 'safe')
        if os.path.exists(safe_dir):
            for filename in os.listdir(safe_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(safe_dir, filename))
                    self.labels.append(1)  # Label 1 for safe

        unsafe_dir = os.path.join(image_dir, 'unsafe')
        if os.path.exists(unsafe_dir):
            for filename in os.listdir(unsafe_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(unsafe_dir, filename))
                    self.labels.append(0)  # Label 0 for unsafe

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ==============================
# Utility: Calculate IoU
# ==============================
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area
    return iou

# ==============================
# Safety Analysis using YOLO & CNN
# ==============================
def analyze_safety_with_cnn(yolo_model, cnn_classifier, image_path, device='cpu'):
    """Run YOLO detection on an image, extract person regions, and classify them with the CNN."""
    results = yolo_model.predict(image_path, conf=0.25)
    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    result_image = original_image_rgb.copy()
    safety_results = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        # Adjust person class index as needed (here assumed to be 5)
        person_indices = np.where(classes == 5)[0]
        for idx in person_indices:
            person_box = boxes[idx]
            x1, y1, x2, y2 = [int(coord) for coord in person_box]
            pad = 20
            x1_padded = max(0, x1 - pad)
            y1_padded = max(0, y1 - pad)
            x2_padded = min(original_image.shape[1], x2 + pad)
            y2_padded = min(original_image.shape[0], y2 + pad)
            person_roi = original_image_rgb[y1_padded:y2_padded, x1_padded:x2_padded]
            if person_roi.size == 0:
                continue
            is_safe, confidence = cnn_classifier.predict_safety(person_roi, device)
            safety_results.append({
                'box': [x1, y1, x2, y2],
                'is_safe': bool(is_safe),
                'confidence': confidence
            })
            color = (0, 255, 0) if is_safe else (255, 0, 0)
            label = f"SAFE: {confidence:.2f}" if is_safe else f"UNSAFE: {confidence:.2f}"
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
            cv2.putText(result_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return result_image, safety_results

# ==============================
# Training function for the CNN classifier
# ==============================
def train_cnn_classifier(model, train_dir, val_dir, epochs=10, batch_size=32, learning_rate=0.001, device='cpu'):
    model.to(device)
    train_dataset = SafetyDataset(train_dir, transform=model.transform)
    val_dataset = SafetyDataset(val_dir, transform=model.transform)
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training or validation dataset is empty. Please check the dataset directories.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_safety_classifier.pth")
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    model.load_state_dict(torch.load("best_safety_classifier.pth"))
    return model

# ==============================
# Generate Safety Training Data using YOLO detections
# ==============================
def generate_safety_training_data(yolo_model, image_dirs, output_dir):
    """
    This function processes images from the YOLO dataset (e.g. css-data)
    and extracts person regions. It then uses a simple heuristic:
      - If an unsafe object (e.g., class 2,3,4) overlaps (IoU > 0.1) with a person,
        the extracted ROI is labeled as unsafe.
      - Otherwise, it is labeled as safe.
    The extracted person ROIs are saved to the output_dir with subfolders 'safe' and 'unsafe'.
    """
    safe_dir = os.path.join(output_dir, 'safe')
    unsafe_dir = os.path.join(output_dir, 'unsafe')
    os.makedirs(safe_dir, exist_ok=True)
    os.makedirs(unsafe_dir, exist_ok=True)
    image_count = 0
    for image_dir in image_dirs:
        for filename in os.listdir(image_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_path = os.path.join(image_dir, filename)
            results = yolo_model.predict(image_path, conf=0.25)
            original_image = cv2.imread(image_path)
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                # Assuming class 5 is 'person'
                person_indices = np.where(classes == 5)[0]
                for idx in person_indices:
                    person_box = boxes[idx]
                    x1, y1, x2, y2 = [int(coord) for coord in person_box]
                    pad = 20
                    x1_padded = max(0, x1 - pad)
                    y1_padded = max(0, y1 - pad)
                    x2_padded = min(original_image.shape[1], x2 + pad)
                    y2_padded = min(original_image.shape[0], y2 + pad)
                    person_roi = original_image_rgb[y1_padded:y2_padded, x1_padded:x2_padded]
                    if person_roi.size == 0:
                        continue
                    # Define unsafe object classes (example: 2, 3, 4)
                    unsafe_objects = [2, 3, 4]
                    has_unsafe = False
                    for class_id in unsafe_objects:
                        if class_id in classes:
                            unsafe_indices = np.where(classes == class_id)[0]
                            for unsafe_idx in unsafe_indices:
                                unsafe_box = boxes[unsafe_idx]
                                if calculate_iou(person_box, unsafe_box) > 0.1:
                                    has_unsafe = True
                                    break
                            if has_unsafe:
                                break
                    save_dir = unsafe_dir if has_unsafe else safe_dir
                    save_path = os.path.join(save_dir, f"person_{image_count:05d}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(person_roi, cv2.COLOR_RGB2BGR))
                    image_count += 1
    print(f"Generated {image_count} training images.")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load your trained YOLOv8 model (update path if necessary)
    yolo_weights_path = "Objectdetection/weights/best_model.pt"
    yolo_model = YOLO(yolo_weights_path)

    # Directories for YOLO dataset images (used to generate CNN training data)
    yolo_image_dirs = ["css-data/train/images"]  # adjust if you have multiple directories

    # Directory to store generated safety training data
    safety_data_dir = "safety_training_data"

    # If training data is not already generated, do so now
    safe_subdir = os.path.join(safety_data_dir, "safe")
    if not os.path.exists(safe_subdir) or len(os.listdir(safe_subdir)) == 0:
        print("Generating safety training data using YOLO detections...")
        generate_safety_training_data(yolo_model, yolo_image_dirs, safety_data_dir)
    else:
        print("Safety training data already exists.")

    # For training, we use the generated data.
    # (For simplicity, here we use the same folder for training and validation.)
    train_data_dir = safety_data_dir
    val_data_dir = safety_data_dir

    # Create and train the CNN safety classifier
    cnn_classifier = PPESafetyClassifier(input_size=128)
    trained_classifier = train_cnn_classifier(cnn_classifier, train_data_dir, val_data_dir,
                                                epochs=10, batch_size=32, learning_rate=0.001, device=device)

    # Run safety analysis on a test image (adjust path as needed)
    test_image = "css-data/test/images/-4405-_png_jpg.rf.82b5c10b2acd1cfaa24259ada8e599fe.jpg"
    result_image, safety_results = analyze_safety_with_cnn(yolo_model, trained_classifier, test_image, device)

    # Display and save the result image
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.axis('off')
    plt.savefig("safety_analysis_result.jpg", bbox_inches='tight')
    plt.show()

    # Print the safety results for each detected person
    for i, result in enumerate(safety_results):
        status = "SAFE" if result["is_safe"] else "UNSAFE"
        print(f"Person {i+1}: {status} (confidence: {result['confidence']:.2f})")
