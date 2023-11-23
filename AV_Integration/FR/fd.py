#----------------------------------------------------------------
# Refer to https://github.com/keyurr2/face-detection/tree/master
#----------------------------------------------------------------

"""

    Created on Sun Dec 2 20:54:11 2018
    
    @author: keyur-r

    SSD pretrained caffe model based face detection using it with opencv's dnn module.
    (https://docs.opencv.org/3.4.0/d5/de7/tutorial_dnn_googlenet.html)
    
    python face_detection_ssd.py -p <prototxt> -m <caffe-model> -t <thresold>

"""

from imutils import face_utils, video
import dlib
import cv2
import argparse
import os
import numpy as np
import math
import sys

if sys.platform == "linux" or sys.platform == "linux2":
    FR_PATH = '/home/jschoi/work/sHRI_base/AV_Integration/FR/' # for Linux
elif sys.platform == "darwin":
    FR_PATH='/Users/jschoi/work/sHRI_base/AV_Integration/FR/' # for macOS 

# Load a pre-trained shape predictor model (68 facial landmarks)
shape_predictor = dlib.shape_predictor(FR_PATH+"shape_predictor_68_face_landmarks_GTX.dat") # https://github.com/davisking/dlib-models

# https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = [
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
]
# Initialize variables for tracking lip motion
lip_points = list(range(48, 68))  # Indices for the lip landmarks
jaw_points = list(range(0,17))  # Indices for the jaw landmarks
face_points = list(range(0,68))
#lip_threshold = 10  # Lip motion threshold
lip_threshold = 20  # Lip motion threshold

def draw_fancy_box(img, pt1, pt2, color, thickness, r, d):
    '''
    To draw some fancy box around founded faces in stream
    '''
    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def visual_VAD(frame, face_roi):
        
    # Detect facial landmarks
    shape = shape_predictor(frame, face_roi)

    # Extract lip landmarks
    lip_landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in lip_points])
    jaw_landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in jaw_points])
    face_landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in face_points])

    landmarks = face_landmarks

    # Calculate lip motion
    lip_motion = np.mean(lip_landmarks[:, 1]) - lip_landmarks[0, 1]
    #print(f"lip_motion: {lip_motion},  lip_threshold: {lip_threshold}")

    if lip_motion > lip_threshold:
        # Lip motion detected, someone has started talking
        voice_activity = True
    else:
        voice_activity = False
    
    return voice_activity, landmarks

def find_faces(img, detections, args):
    total_faces = 0
    detected_l = []
    # Draw boxes around found faces
    for i in range(0, detections.shape[2]):
        # Probability of prediction
        prediction_score = detections[0, 0, i, 2]
        if prediction_score < args.thresold:
            continue
        # Finding height and width of frame
        (h, w) = img.shape[:2]
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        total_faces += 1

        prediction_score_str = "{:.2f}%".format(prediction_score * 100)

        label = "Face #{} ({})".format(total_faces, prediction_score_str)

        # https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
        draw_fancy_box(img, (x1, y1), (x2, y2), (127, 255, 255), 2, 10, 20)
        # show the face number with prediction score
        cv2.putText(img, label, (x1 - 20, y1 - 20),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, (51, 51, 255), 2)
        
        detected = { 'box': (x1,x2,y1,y2)}
        detected_l.append(detected)


        #cv2.circle(img, (img_center_x, img_center_y), 10, (0,255,0),5) 

        #print(f'center: {frame_center_x}, width: {width}')
        #print(f'ratio: {ratio}')      
        #print(f'img_ang: {img_ang*180/math.pi}')

    return img, total_faces, detected_l

    # show the output frame
    #cv2.imshow("Face Detection with SSD", img)


def face_detection_realtime(detector, args):

    # Feed from computer camera with threading
    camera_idx = 0
    #camera_idx = 1
    cap = video.VideoStream(camera_idx).start()


    MAX_ANGLE = 45*math.pi/180
    tan_MAX_ANGLE = math.tan(MAX_ANGLE)

    global g_fd_results

    while True:

        # Getting out image frame by webcam
        img = cap.read()

        # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
        inputBlob = cv2.dnn.blobFromImage(cv2.resize(
            img, (300, 300)), 1, (300, 300), (104, 177, 123))

        detector.setInput(inputBlob)
        detections = detector.forward()
        img, total_faces, detected_l = find_faces(img, detections, args)

        id = 0
        g_fd_results = []
        for detected in detected_l:
            (x1,x2,y1,y2) = detected['box']

            #face_roi = img[y1:y2, x1:x2]
            face_roi = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
            #print(face_roi)

            v_VAD, landmarks = visual_VAD(img, face_roi)
            for landmark in landmarks:
                cv2.circle(img, (landmark[0], landmark[1]), 4, (0,255,0), 2)


            #width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            (h, w) = img.shape[:2]
            width = w
            img_center_x = int((x1 + x2)/2)
            img_center_y = int((y1 + y2)/2)
            frame_center_x = width/2
            ratio = frame_center_x - img_center_x
            img_ang = math.atan(ratio/frame_center_x*tan_MAX_ANGLE)
            detected['azimuth'] = img_ang
            g_fd_results.append(detected)
            print(f'id: {id}, img_ang: {img_ang*180/math.pi},  detected_boxs: {(x1,x2,y1,y2)}')                  
            id = id + 1
            cv2.circle(img, (img_center_x, img_center_y), 10, (0,255,0),5)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        cv2.imshow("Face Detection with SSD", img)

    cv2.destroyAllWindows()
    cap.stop()

if __name__ == "__main__":

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default="./deploy.prototxt",
                    help="Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="./res10_300x300_ssd_iter_140000.caffemodel",
                    help="Pre-trained caffe model")
    ap.add_argument("-t", "--thresold", type=float, default=0.6,
                    help="Thresold value to filter weak detections")
    args = ap.parse_args()

    # This is based on SSD deep learning pretrained model
    detector = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    print("Real time face detection is starting ... ")
    face_detection_realtime(detector, args)