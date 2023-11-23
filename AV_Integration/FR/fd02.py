#----------------------------------------------------------------
# Refer to https://github.com/keyurr2/face-detection/tree/master
#----------------------------------------------------------------

import cv2
import numpy as np  # Import numpy

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Load the pre-trained face detection model from OpenCV (SSD)
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Resize the frame for better performance
    frame = cv2.resize(frame, (300, 300))

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))

    # Pass the blob through the network to detect faces
    net.setInput(blob)
    detections = net.forward()

    # Loop through the detected faces and draw rectangles
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Adjust the confidence threshold as needed
            box = detections[0, 0, i, 3:7] * np.array([300, 300, 300, 300])
            (startX, startY, endX, endY) = box.astype(int)

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()