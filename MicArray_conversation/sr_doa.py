#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

# To run
# 1. In one terminal, run
#    $ python sr_doa.py 2>/dev/null  # Here '2>/dev/null' is used to disable ALSA warning messages
#
# 2. In the other terminal, move to the folder where odaslive execution exists
#     (~/work/sHRI_base/MicArray_conversion/odas/build/bin)
#    Then, run
#   $ ./odaslive -c ~/work/sHRI_base/MicArray_conversation/odas/odas.cfg 

import sys
import os

# For ssl
import socket
import json
import threading
from threading import Lock
import math


# For stt
import speech_recognition as sr
global BASE_PATH

if sys.platform == "linux" or sys.platform == "linux2":
    BASE_PATH = '/home/jschoi/work/sHRI_base/conversation/' # for Linux
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/home/jschoi/work/sHRI_base/MicArray_conversation/STT/cjsstt.json'
#    os.environ["PYTHONPATH"] = '/home/jschoi/work/sHRI_base:$PYTHONPATH'
elif sys.platform == "darwin":
    BASE_PATH='/Users/jschoi/work/sHRI_base/conversation/' # for macOS 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/Users/jschoi/work/sHRI_base/MicArray_conversation/STT/cjsstt.json'
#    os.environ["PYTHONPATH"] = '/Users/jschoi/work/sHRI_base:$PYTHONPATH'
import STT.gcs_stt as gcs_stt
from google.cloud import speech
import re

# For fd
import FR.fd as fd
import argparse
import cv2
from imutils import face_utils, video


from tuning import Tuning
import usb.core
import usb.util
import time

# For text_classification
from text_classification import MachineLearning

global g_speech_recognized, g_speech_result

sst_az_list = []
sst_az_stream = []
lock_for_sst_az_list = Lock()
verbose = 0   # 0 to disable print in process_ssl_sst_result,   1 to enable it

def process_ssl_sst_result(result_str):
    # Parse the JSON data to extract relevant information
    try:
        global sst_az_list
        global sst_az_stream
        global verbose

        data_stream = json.loads(result_str)

        if "src" in data_stream:
            data = data_stream["src"]
            #print(data)
            #print(f"--- # of sources: {len(data)} ---")

            # json example of 'ssl':
            #{
            #    "timeStamp": 285,
            #    "src": [
            #        { "x": -0.082, "y": -0.957, "z": 0.279, "E": 0.216 },
            #        { "x": 0.196, "y": 0.064, "z": 0.979, "E": 0.131 },
            #        { "x": 0.844, "y": 0.523, "z": 0.117, "E": 0.039 },
            #        { "x": 0.196, "y": 0.847, "z": 0.495, "E": 0.009 }
            #    ]
            #}

            # json example of 'sst':
            #{
            #    "timeStamp": 1071,
            #    "src": [
            #        { "id": 0, "tag": "", "x": 0.000, "y": 0.000, "z": 0.000, "activity": 0.000 },
            #        { "id": 0, "tag": "", "x": 0.000, "y": 0.000, "z": 0.000, "activity": 0.000 },
            #        { "id": 0, "tag": "", "x": 0.000, "y": 0.000, "z": 0.000, "activity": 0.000 },
            #        { "id": 0, "tag": "", "x": 0.000, "y": 0.000, "z": 0.000, "activity": 0.000 }
            #    ]
            #}

            ENERGY_TH = 0.3
            if 'E' in data[0]: # ssl
                if verbose:
                    print("      --- ssl ---")
                for id in range(len(data)):
                    azimuth = data[id]['x']
                    elevation = data[id]['y']
                    energy = data[id]['E']
                    if energy > ENERGY_TH:
                        if verbose:
                            print(f"       ssl: Source detected at azimuth: {azimuth}, elevation: {elevation}")                
            elif 'activity' in data[0]:    # sst
                if data[0]['activity'] > 0:
                    if verbose:
                        print("   #------ sst : start ------#")

                    lock_for_sst_az_list.acquire() 
                    sst_az_list = data
                    sst_az_stream = data_stream
                    lock_for_sst_az_list.release()

                    for id in range(len(data)):
                        if verbose:
                            print(f"   sst: {data[id]}")
                        data_x = data[id]['x']
                        data_y = data[id]['y']
                        azimuth = math.atan2(data_y, data_x) * 180 / math.pi
                        if verbose:
                            print("     azimuth: {:.1f} degree".format(azimuth))
                    if verbose:
                        print("   #------ sst : end ------#")
            else:
                if verbose:
                    print("No valid data found")

    except Exception as e:
        temp = 1
        #print("json load error with ")
        #print(e)


def launch_socket_server(ip, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((ip, port))
            server_socket.listen()
            print(f"Server listening on {ip}:{port}")

            while True:
                client_socket, client_addr = server_socket.accept()
                print(f"Accepted connection from {client_addr}")

                while True:
                    data = client_socket.recv(4096).decode().strip()
                    if not data:
                        break
                    #process_ssl_result(data) 
                    process_ssl_sst_result(data)    

    except KeyboardInterrupt:
        print("Server shutdown.")


def listen_print_loop(responses):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    global g_speech_result, g_speech_recognized

    num_chars_printed = 0
    transcript = "NULL"
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite_chars)
            g_speech_result = 1                 # global
            g_speech_recognized = transcript    # global

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0


def speech_recog():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'ko-KR'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=gcs_stt.RATE,
        language_code=language_code)
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True)
    
    print("Say something!")
    with gcs_stt.MicrophoneStream(gcs_stt.RATE, gcs_stt.CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use. 
        listen_print_loop(responses)


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
        img, total_faces, detected_l = fd.find_faces(img, detections, args)

        id = 0
        g_fd_results = []
        for detected in detected_l:
            (x1,x2,y1,y2) = detected['box']

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

    #------------------------------------------
    # Initialize for face detection

    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", default="./FR/deploy.prototxt",
                    help="Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", default="./FR/res10_300x300_ssd_iter_140000.caffemodel",
                    help="Pre-trained caffe model")
    ap.add_argument("-t", "--thresold", type=float, default=0.6,
                    help="Thresold value to filter weak detections")
    args = ap.parse_args()

    # This is based on SSD deep learning pretrained model
    detector = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    #print("Real time face detection is starting ... ")
    #face_detection_realtime(detector, args)

    #thread_fd = threading.Thread(target=fd.face_detection_realtime, args=(detector, args))
    #thread_fd.start()

    # Feed from computer camera with threading
    camera_idx = 0
    #camera_idx = 1
    cap = video.VideoStream(camera_idx).start()

    MAX_ANGLE = 45*math.pi/180
    tan_MAX_ANGLE = math.tan(MAX_ANGLE)

    global g_fd_results
    #------------------------------------------



    # Replace "ODAS_SERVER_IP" and "ODAS_SERVER_PORT" with the desired IP and port for the server
    #odas_server_ip = "192.168.1.6"
    #odas_server_ip = "192.168.1.21"
    #odas_server_ip = socket.gethostbyname(socket.gethostname())

    # Get ip address of host
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8",80))
    odas_server_ip = s.getsockname()[0]
    odas_server_ssl_port = 9001
    odas_server_sst_port = 9000

    #server_thread_ssl = threading.Thread(target=launch_socket_server, args=(odas_server_ip, odas_server_ssl_port))
    #server_thread_ssl.start()

    server_thread_sst = threading.Thread(target=launch_socket_server, args=(odas_server_ip, odas_server_sst_port))
    server_thread_sst.start()

    thread_gcs_stt = threading.Thread(target=speech_recog)
    thread_gcs_stt.start()

    #dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    #Mic_tuning = Tuning(dev)
    #print(Mic_tuning.direction)

    # obtain audio from the microphone
    r = sr.Recognizer()

    # recognize speech using Google Cloud Speech
    #GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""INSERT THE CONTENTS OF THE GOOGLE CLOUD SPEECH JSON CREDENTIALS FILE HERE"""
    #GOOGLE_CLOUD_SPEECH_CREDENTIALS = "/home/jschoi/work/sHRI_base/STT/cjsstt.json"


    # text_classification
    #BASE_PATH='/home/jschoi/work/sHRI_base/conversation/'
    ml = MachineLearning(BASE_PATH)

    bOnline = True
    #test_conversations = []
    #test_conversations.append("그래. 무슨일이지?")
    #test_labels = []
    #for id in range(len(test_conversations)):
    #    test_labels.append(0)   # forcing 0s because it's not important.

    #ml.test(test_conversations, test_labels, bOnline)

    n_prevTimeStamp = 0
    g_speech_result = 0
    g_speech_recognized = "NULL"
    while True:
        try:
            #---------------------------
            #--- face detection - start

            # Getting out image frame by webcam
            img = cap.read()

            # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
            inputBlob = cv2.dnn.blobFromImage(cv2.resize(
                img, (300, 300)), 1, (300, 300), (104, 177, 123))

            detector.setInput(inputBlob)
            detections = detector.forward()
            img, total_faces, detected_l = fd.find_faces(img, detections, args)

            id = 0
            g_fd_results = []
            for detected in detected_l:
                (x1,x2,y1,y2) = detected['box']

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
                #print(f'id: {id}, img_ang: {img_ang*180/math.pi},  detected_boxs: {(x1,x2,y1,y2)}')                  
                id = id + 1
                cv2.circle(img, (img_center_x, img_center_y), 10, (0,255,0),5)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            cv2.imshow("Face Detection with SSD", img)
            #--- face detection - end
            #---------------------------



            '''
            with sr.Microphone() as source:
                #r.adjust_for_ambient_noise(source)
                print("Say something!")
                audio = r.listen(source)

            test_conversations = []

            # Google Cloud Speech-to-Text
            test_conversations.append(r.recognize_google_cloud(audio, language="ko-KR", credentials_json=os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')))

            # Google web speech API
            #test_conversations.append(r.recognize_google(audio, language="ko-KR"))
            '''

            if g_speech_result:
                g_speech_result = 0

                test_conversations = []
                test_conversations.append(g_speech_recognized)
            
                test_labels = []
                test_labels.append(0)
                #print("Google Cloud Speech thinks you said " + test_conversations[0])
                #print("Speaker Direction : {}".format(Mic_tuning.direction))


                classification = ml.test(test_conversations, test_labels, bOnline)
                id = 0
                print(" #---- sc ----#")
                if classification[id] == 1:
                    print(f"   [{classification[id] }]: SENIOR! \n")
                elif classification[id] == 0:
                    print(f"   [{classification[id]}]: JUNIOR! \n")
                else:
                    print(f"   [{classification[id]}]: NOT DETERMINED! \n")


                #print(sst_az_list)
                data = sst_az_list
                
                if len(data) > 0 and n_prevTimeStamp != sst_az_stream['timeStamp']:
                    print(f" n_prevTimeStamp: {n_prevTimeStamp}, current timeStamp: {sst_az_stream['timeStamp']}")
                    n_prevTimeStamp = sst_az_stream['timeStamp']
                    if 'activity' in data[0]:    # sst
                        if data[0]['activity'] > 0:
                            print(" #---- sst ----#")

                            for id in range(len(data)):
                                if data[id]['activity'] > 0:
                                    print(f"   sst: {data[id]}")
                                    data_x = data[id]['x']
                                    data_y = data[id]['y']
                                    azimuth = math.atan2(data_y, data_x) * 180 / math.pi
                                    print("   azimuth: {:.1f} degree".format(azimuth))

                                    id=0
                                    for detected in detected_l:
                                        (x1,x2,y1,y2) = detected['box']
                                        img_ang = detected['azimuth']
                                        print(f'id: {id}, img_ang: {img_ang*180/math.pi},  detected_boxs: {(x1,x2,y1,y2)}') 
                                        id = id + 1


                print("\n")
                print("Say something!")
                    

        except sr.UnknownValueError:
            print("Google Cloud Speech could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Cloud Speech service; {0}".format(e))


    cv2.destroyAllWindows()
    cap.stop()        
