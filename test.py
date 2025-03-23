
from preprocess import *

def real_time_detection(model, camera_id=0):
    """Run real-time detection using webcam."""
    cap = cv2.VideoCapture(camera_id)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame.")
            break
        
        # Run inference
        results = model.predict(frame, conf=0.25)
        result = results[0]
        
        # Visualize results on frame
        annotated_frame = result.plot()
        
        # Display the frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        
        # Quit with 'q'
        if cv2.waitKey(1) == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()


