#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

# To run
# 1. In one terminal, run
#    $ python sr_doa.py 2>/dev/null  # Here '2>/dev/null' is used to disable ALSA warning messages
#
# 2. Then, run
#   $ ./odas/build/bin/odaslive -c ./odas/odas.cfg 

import sys
import os
import cv2
import numpy as np

# For ssl
import math


if sys.platform == "linux" or sys.platform == "linux2":
    BASE_PATH = '/home/jschoi/work/sHRI_base/conversation/' # for Linux
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/jschoi/work/sHRI_base/AV_Integration/STT/cjsstt.json'
#    os.environ["PYTHONPATH"] = '/home/jschoi/work/sHRI_base:$PYTHONPATH'
elif sys.platform == "darwin":
    BASE_PATH='/Users/jschoi/work/sHRI_base/conversation/' # for macOS 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/jschoi/work/sHRI_base/AV_Integration/STT/cjsstt.json'
#    os.environ["PYTHONPATH"] = '/Users/jschoi/work/sHRI_base:$PYTHONPATH'

#---------------------------------------
# For Face Interaction (FI)
# ----------------------------
# Head Pose detection: by Dlib:
#   hp_detection.py
#   You can download a trained facial shape predictor from:
#    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
from hp_detection import HeadPose

# ----------------------------
# Face Recognition: by Dlib:
#    face_recognition/fr_dlib.py
#  ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
#    You can download a trained facial shape predictor and recognition model from:\n"
#     http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
#     http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
from fr_dlib import FaceRecog

# ------------------------
# Object Tracking by Dlib correlation_tracker
from obj_tracker import Obj_Tracker

# ----------------------------
#   Action Event Detection: action_detection/action_detection.py
from action_detect import Event_Detector

class FI():  # Face Interaction Class
    def __init__(self, path, cap):
        cap.set(3, 320)
        cap.set(4, 240)

        ret, sample_frame = cap.read()

        self.path = path
        self.cap = cap
        # ----------------------------
        # Head Pose Detection: by Dlib
        predictor_path = path + "/shape_predictor_68_face_landmarks_GTX.dat"
        #predictor = dlib.shape_predictor(predictor_path)
        self.hpd = HeadPose(sample_frame, path, predictor_path)

        # ------------------------------------------
        # Dlib: Load labels_id & face_descriptors of registered faces
        predictor_path = path + "/shape_predictor_5_face_landmarks.dat"
        face_rec_model_path = path + "/dlib_face_recognition_resnet_model_v1.dat"
        self.fr = FaceRecog(predictor_path, face_rec_model_path, fr_th=0.5)
        self.iter = 0
        # ------------------------------------------

        # ------------------------------------------
        # Generate a class for event detection such as approach or disappear
        self.event_detect = Event_Detector()

        # ------------------------
        # Object Tracking by Dlib correlation_tracker
        self.obj_track = Obj_Tracker()

        # ------------------------
        # Object Tracking by Dlib correlation_tracker
        self.obj_track = Obj_Tracker()

        self.max_width = 0  #frame.shape[0]
        self.max_width_id = -1
        self.fr_labels = []
        self.fr_box = []
        self.fr_min_dist = 0
        self.ad_state = []
        self.ad_event = []

    def run(self):
        cap = self.cap

        hpd = self.hpd
        fr = self.fr
        event_detect = self.event_detect
        obj_track = self.obj_track

        iter = self.iter
        max_width = self.max_width
        max_width_id = self.max_width_id
        fr_labels = self.fr_labels
        fr_box = self.fr_box
        fr_min_dist = self.fr_min_dist
        ad_state = self.ad_state
        ad_event = self.ad_event
    
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #-------------------------------------
        #  일정시간마다 tracking reset하기
        if iter % 100 == 0:
            obj_track.track_started = False
            if obj_track.track_started == True:
                if len(fr_labels) > 0 and fr_labels[max_width_id] == "unknown_far":
                    event_detect.reset()
        iter += 1

        # 아니면, 매번 얼굴인식 수행
        # Face Recognition
        #(fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)
        #-------------------------------------


        #-------------------------------------
        #  일정시간마다 tracking reset하기
        if iter % 100 == 0:
            obj_track.track_started = False
            if obj_track.track_started == True:
                if len(fr_labels) > 0 and fr_labels[max_width_id] == "unknown_far":
                    event_detect.reset()
        iter += 1

        # 아니면, 매번 얼굴인식 수행
        # Face Recognition
        #(fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)
        #-------------------------------------

        # --------------------------------------
        # Object Tracking for undetected face
        #   Ref: https://www.codesofinterest.com/2018/02/track-any-object-in-video-with-dlib.html
        if obj_track.track_started == False:
            # ---------------------------------
            # Face Recognition
            (fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)

            if len(fr_labels) > 0:
                obj_track.track_started = True
                obj_track.start_tracking(frame, fr_box[max_width_id])
                obj_track.label = fr_labels[max_width_id]
                obj_track.tracking(frame)
                obj_track.min_dist = fr_min_dist[max_width_id]

        else:
            if len(fr_labels) > 0 and fr_labels[max_width_id] == "unknown_far":
                event_detect.reset()

                # ---------------------------------
                # Face Recognition
                (fr_labels, fr_box, fr_min_dist) = fr.face_recognition(frame)

                if len(fr_labels) > 0:
                    obj_track.track_started = True
                    obj_track.start_tracking(frame, fr_box[max_width_id])
                    obj_track.label = fr_labels[max_width_id]
                    obj_track.tracking(frame)
            else:
                max_width_id = 0
                fr_labels = []
                fr_box = []
                fr_min_dist = []
                fr_labels.append(obj_track.label)
                fr_box.append(obj_track.roi)
                fr_min_dist.append(obj_track.min_dist)

                obj_track.tracking(frame)
                if obj_track.track_started == False:
                    fr_labels = []
                    fr_box = []
        # --------------------------------------


        # --------------------------------------
        # Display for the name of the selected face
        for id in range(len(fr_labels)):
            selected_label = fr_labels[id]
            d = fr_box[id]
            min_dist = fr_min_dist[id]

            if(selected_label != None):
                #print(selected_label)
                font = cv2.FONT_HERSHEY_SIMPLEX
                conf_str = "{0:3.1f}".format(min_dist)
                name = selected_label + ", [" + conf_str + "]"
                color = (255, 255, 255)
                stroke = 1   # 글씨 굵기 ?
                cv2.putText(frame, name, (d.left(), d.top()), font, 0.5, color, stroke, cv2.LINE_AA)


            # ---------------------------------
            #   Select the closest face
            d_width = d.right() - d.left()
            if(d_width > max_width):
                max_width_id = id
                max_width = d_width


        if(len(fr_labels) > 0):
            # ---------------------------------
            # Head Pose Detection for the closest face,

            # Get the landmarks/parts for the face in box d.
            d = fr_box[max_width_id]
            shape = hpd.predictor(frame, d)    # predict 68_face_landmarks
            #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))

            # Find Head Pose using the face landmarks and Draw them on the screen.
            (p1, p2) = hpd.draw_landmark_headpose(frame, shape)
            roi_ratio = (d.right() - d.left()) / frame.shape[0]

            dist = np.subtract(p2, p1)
            dist = np.sqrt(np.dot(dist, dist))
            dist_ratio = dist / (d.right() - d.left())

            roi_ratio_th = 0.15
            dist_ratio_th = 0.75  # 0.03
            #print(" ")
            #print("roi_ratio: %3.2f, dist_ratio: %5.4f" % (roi_ratio, dist_ratio))
            if roi_ratio > roi_ratio_th and dist_ratio < dist_ratio_th:
                cv2.line(frame, p1, p2, (0, 0, 255), 2)
            else:
                cv2.line(frame, p1, p2, (0, 255, 0), 2)


        (ad_state, ad_event) = event_detect.approach_disappear(fr_labels, fr_box, max_width_id)

        self.iter = iter
        self.fr_labels = fr_labels
        self.fr_box = fr_box
        self.fr_min_dist = fr_min_dist
        self.max_max_width = max_width
        self.max_width_id = max_width_id
        self.ad_state = ad_state
        self.ad_event = ad_event

        return frame



if __name__ == "__main__":

    camera_idx = 0
    cap = cv2.VideoCapture(camera_idx)

    fi = FI(cap)    # Face Interaction Class


    while True:

        #---------------------------
        #--- face detection
        # Capture frame-by-frame
        frame = fi.run()
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        cv2.imshow("Face Detection with SSD", frame)
        #---------------------------
            

    cv2.destroyAllWindows()
    cap.stop()   
     
