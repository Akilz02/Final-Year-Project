from ultralytics import YOLO
import cv2

# Load your trained model (update the path to your actual model file)
# model = YOLO('C:\Users\ASUS\Desktop\APIIT\FYP\YOLOV11\model_- 29 may 2025 10_31.pt')
model = YOLO('C:/Users/ASUS/Desktop/APIIT/FYP/YOLOV11/model_- 29 may 2025 10_31.pt')

# Start webcam (0 is default camera)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Plot results on the frame
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow("YOLO Inference", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
