import cv2
import torch
import numpy as np
import datetime
from models.common import DetectMultiBackend, AutoShape
import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

#Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config value
video_path = "testfinal.mp4"
conf_threshold = 0.5
vehicles_class = [0]

# Load the pre-trained YOLOv9 model
model  = DetectMultiBackend(weights="weights/best.pt", fuse=True, device=device)
model  = AutoShape(model)

# Load classname
with open("data_test/classes.names") as f:
    class_names = f.read().strip().split('\n')

colors = np.random.randint(0,255, size=(len(class_names),3 ))
tracks = []

# Initialize the video capture object
cap = cv2.VideoCapture(video_path)

while True:
    start = datetime.datetime.now()
    ret, frame = cap.read()
    if not ret:
        continue
    # Run the YOLO model on the frame
    results = model(frame)

    ######################################
    # DETECTION
    ######################################

    # Loop over the results
    for detect_object in results.pred[0]:
        # Extract the label, confidence, bounding box associated with the prediction
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        # Check if class_id in vehicles_class and confidence greater than conf_threshold
        if vehicles_class is None:
            if confidence < conf_threshold:
                continue
        else:
            if class_id not in vehicles_class or confidence < conf_threshold:
                continue

    ######################################
    # OCR
    ######################################

        color = colors[class_id]
        B, G, R = map(int,color)

        # Crop license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

        # Process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        # Using Adaptive Thresholding
        # license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        # license_plate_crop_thresh = cv2.adaptiveThreshold(license_plate_crop_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        # read license plate number
        license_plate_text = reader.readtext(license_plate_crop_thresh, paragraph=True, detail = 0)
        license_plate_text = license_plate_text[0].replace(' ', '').replace('_', '')
        label = "{}:{:.2f}".format(class_names[class_id], confidence)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 1, (204, 0, 20), 2)
        cv2.putText(frame, license_plate_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

    # End time to compute the fps
    end = datetime.datetime.now()
    # Show the time it took to process 1 frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")

    # Show the frame to our screen
    cv2.imshow('Gray image', license_plate_crop_gray)
    cv2.imshow('thresh image', license_plate_crop_thresh)
    cv2.imshow("Vehicles Tracking", frame)
    # Enter "Q" for break
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
