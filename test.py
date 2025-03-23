import cv2
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
from rcnn import PPESafetyClassifier  # Ensure this import matches your module structure
# from preprocess import *  # Include if you have additional preprocessing functions

def real_time_detection_with_classification(yolo_model, cnn_classifier, camera_id=0, device='cpu'):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get class names from YOLO model if available (e.g. model.names)
    class_names = yolo_model.names if hasattr(yolo_model, "names") else {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame.")
            break

        # Run YOLO inference on the current frame
        results = yolo_model.predict(frame, conf=0.25)
        annotated_frame = frame.copy()

        # Process each detection result from YOLO
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            classes = result.boxes.cls.cpu().numpy()   # Class indices
            confidences = result.boxes.conf.cpu().numpy()  # YOLO confidence scores

            # Iterate over all detections
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                # Default YOLO label
                yolo_label = f"{class_names.get(int(cls), int(cls))}: {conf:.2f}" if class_names else f"Class {int(cls)}: {conf:.2f}"
                
                # If the object is a person (class index 5), run safety classification
                if int(cls) == 5:
                    pad = 20
                    x1_padded = max(0, x1 - pad)
                    y1_padded = max(0, y1 - pad)
                    x2_padded = min(frame.shape[1], x2 + pad)
                    y2_padded = min(frame.shape[0], y2 + pad)
                    person_roi = frame[y1_padded:y2_padded, x1_padded:x2_padded]

                    # Run CNN classifier on the person ROI
                    prediction, safety_conf = cnn_classifier.predict_safety(person_roi, device=device)
                    safety_label = "SAFE" if prediction == 1 else "UNSAFE"
                    # Append safety info to the YOLO label
                    label = f"{yolo_label} | {safety_label}: {safety_conf:.2f}"
                    color = (0, 255, 0) if prediction == 1 else (0, 0, 255)
                else:
                    label = yolo_label
                    color = (255, 255, 0)  # For non-person objects (example color)

                # Draw the bounding box and label on the annotated frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Real-Time Detection & Classification", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load YOLO model and CNN classifier
    yolo_model = YOLO("Objectdetection/weights/best_model.pt")  # Ensure this path is correct
    cnn_classifier = PPESafetyClassifier(input_size=128)
    cnn_model_path = "Objectdetection/best_safety_classifier.pth"  # Update path if needed
    cnn_classifier.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))
    cnn_classifier.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_classifier.to(device)

    # Run real-time detection with integrated classification
    real_time_detection_with_classification(yolo_model, cnn_classifier, camera_id=0, device=device)
