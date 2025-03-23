
from ultralytics.data.utils import check_det_dataset
import albumentations as A
import cv2
import numpy as np
import glob
import os

def enhance_images(input_folder, output_folder):
    """Apply CLAHE enhancement to images recursively."""
    os.makedirs(output_folder, exist_ok=True)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Use recursive glob pattern
    image_paths = glob.glob(os.path.join(input_folder, "**", "*.jpg"), recursive=True)
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Convert to LAB, apply CLAHE, and convert back
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        # Preserve subfolder structure
        relative_path = os.path.relpath(img_path, input_folder)
        output_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cv2.imwrite(output_path, enhanced_img)
    
    print(f"Enhanced {len(image_paths)} images")


def create_custom_augmentations():
    """Create a pipeline of augmentations for training data."""
    transform = A.Compose([
        A.RandomRotate90(),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.3),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    return transform

# This function can be used to apply custom augmentations to images
def apply_augmentations(image_path, label_path, transform):
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read YOLO labels
    with open(label_path, 'r') as f:
        labels = f.readlines()
    
    bboxes = []
    class_labels = []
    
    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        bboxes.append([x_center, y_center, width, height])
        class_labels.append(class_id)
    
    # Apply augmentation
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    return transformed['image'], transformed['bboxes'], transformed['class_labels']


if __name__ == "__main__":
    input_folder = "/home/demo/MachineVisionProject/css-data"
    output_folder = "/home/demo/MachineVisionProject/enhanced_images"

    print(f"Enhancing images from {input_folder} and saving to {output_folder}...")
    enhance_images(input_folder, output_folder)
    print("Enhancement completed.")

