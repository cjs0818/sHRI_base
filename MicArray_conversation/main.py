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
global g_fd_results, n_prevTimeStamp
global g_ssl_results
global g_sst_az_list, g_sst_az_stream
global g_start_time

g_sst_az_list = []
g_sst_az_stream = []
lock_for_g_sst_az_list = Lock()
verbose = 0   # 0 to disable print in process_ssl_sst_result,   1 to enable it

global g_human


MAX_ANGLE = 45*math.pi/180
#MAX_ANGLE = 15*math.pi/180
tan_MAX_ANGLE = math.tan(MAX_ANGLE)
ANG_DIFF_TH = 20*math.pi/180
#ANG_DIFF_TH = 10*math.pi/180

def get_current_time() -> int:
    """Return Current Time in MS.

    Returns:
        int: Current Time in MS.
    """
    return int(round(time.time() * 1000))

def process_ssl_sst_result(result_str):
    # Parse the JSON data to extract relevant information
    try:
        global g_sst_az_list
        global g_sst_az_stream
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

                    lock_for_g_sst_az_list.acquire() 
                    g_sst_az_list = data
                    g_sst_az_stream = data_stream
                    lock_for_g_sst_az_list.release()

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


def listen_print_loop(responses, stream):
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
        if stream.get_current_time() - stream.start_time > gcs_stt.STREAMING_LIMIT:
            stream.start_time = stream.get_current_time()
            break

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


        result_seconds = 0
        result_micros = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.microseconds:
            result_micros = result.result_end_time.microseconds

        stream.result_end_time = int((result_seconds * 1000) + (result_micros / 1000))

        corrected_time = (
            stream.result_end_time
            - stream.bridging_offset
            + (gcs_stt.STREAMING_LIMIT * stream.restart_counter)
        )


        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            #sys.stdout.write(gcs_stt.GREEN)
            #sys.stdout.write("\033[K")
        
            sys.stdout.write(transcript + overwrite_chars + '\r')
            sys.stdout.flush()

            num_chars_printed = len(transcript)
            stream.last_transcript_was_final = False
        else:
            #sys.stdout.write(gcs_stt.RED)
            #sys.stdout.write("\033[K")
        
            print(transcript + overwrite_chars)
            #sys.stdout.write(gcs_stt.RESET)
            #sys.stdout.write("\033[K")
            g_speech_result = 1                 # global
            g_speech_recognized = transcript    # global

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

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
        model="latest_long",    # To speed up the response of 'is_final'
        language_code=language_code)
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True)
    
    print("Say something!")
    with gcs_stt.MicrophoneStream(gcs_stt.RATE, gcs_stt.CHUNK) as stream:
        while not stream.closed:
            sys.stdout.write(gcs_stt.YELLOW)
            sys.stdout.write(
                "\n" + str(gcs_stt.STREAMING_LIMIT * stream.restart_counter) + ": NEW REQUEST\n"
            )
            sys.stdout.write(gcs_stt.RESET)
            stream.audio_input = []
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            responses = client.streaming_recognize(streaming_config, requests)

            # Now, put the transcription responses to use. 
            listen_print_loop(responses, stream)

            if stream.result_end_time > 0: 
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write("\n")
            stream.new_stream = True


def init_fd(camera_idx):
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
    fd_detector = cv2.dnn.readNetFromCaffe(args.prototxt, args.model)

    #thread_fd = threading.Thread(target=fd.face_detection_realtime, args=(detector, args))
    #thread_fd.start()

    cap = video.VideoStream(camera_idx).start()

    return fd_detector, cap, args
    #------------------------------------------

def face_detection(detector, cap, args):
    global g_fd_results, g_ssl_results

    #---------------------------
    #--- face detection - start

    # Getting out image frame by webcam
    detected_l = []
    
    img = cap.read()

    # https://docs.opencv.org/trunk/d6/d0f/group__dnn.html#ga29f34df9376379a603acd8df581ac8d7
    inputBlob = cv2.dnn.blobFromImage(cv2.resize(
        img, (300, 300)), 1, (300, 300), (104, 177, 123))

    detector.setInput(inputBlob)
    detections = detector.forward()
    img, total_faces, detected_l = fd.find_faces(img, detections, args)

    id = 0
    g_fd_results = []
    color_green = (0,255,0)
    color_red = (0,0,255)
    for detected in detected_l:
        (x1,x2,y1,y2) = detected['box']

        #face_roi = img[y1:y2, x1:x2]
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

        color = color_green
        #print(g_ssl_results)
        for ssl_result in g_ssl_results:
            ssl_azimuth = ssl_result['azimuth']
            ang_diff = math.fabs(ssl_azimuth - img_ang)
            #print(f'ang_diff: {ang_diff*180/math.pi}, ANG_DIFF_TH: {ANG_DIFF_TH*180/math.pi}')
            time_diff = get_current_time() - ssl_result['time']

            TIME_DIFF_TH = 5000 # milisecond
            if ang_diff < ANG_DIFF_TH and time_diff < TIME_DIFF_TH:
                color = color_red

        cv2.circle(img, (img_center_x, img_center_y), 10, color,5)

    return img



def sst_check():
    global g_fd_results, n_prevTimeStamp
    global g_sst_az_stream, g_sst_az_list

    #print(g_sst_az_list)
    data = g_sst_az_list
    
    ssl_results = []
    if len(data) > 0 and n_prevTimeStamp != g_sst_az_stream['timeStamp']:
        print(f" n_prevTimeStamp: {n_prevTimeStamp}, current timeStamp: {g_sst_az_stream['timeStamp']}")
        n_prevTimeStamp = g_sst_az_stream['timeStamp']
        if 'activity' in data[0]:    # sst
            if data[0]['activity'] > 0:
                print(" #---- sst ----#")

                for id in range(len(data)):
                    if data[id]['activity'] > 0:
                        print(f"   sst: {data[id]}")
                        data_x = data[id]['x']
                        data_y = data[id]['y']
                        azimuth = math.atan2(data_y, data_x)
                        ssl_result = {'azimuth': azimuth, 'time': get_current_time()}
                        ssl_results.append(ssl_result)
                        
                        sys.stdout.write(gcs_stt.GREEN)
                        sys.stdout.write("\033[K")
                        print("   azimuth: {:.1f} degree".format(azimuth * 180/math.pi))
                        sys.stdout.write(gcs_stt.RESET)
                        sys.stdout.write("\033[K")

                        id=0
                        for detected in g_fd_results:
                            (x1,x2,y1,y2) = detected['box']
                            img_ang = detected['azimuth']
                            #print(f'      id: {id}, img_ang: {img_ang*180/math.pi},  detected_boxs: {(x1,x2,y1,y2)}') 
                            print(f'      id: {id}, img_ang: {img_ang*180/math.pi}') 
                            id = id + 1
    return ssl_results


if __name__ == "__main__":

    #-------------------------------------
    # Initialize
    global g_speech_recognized, g_speech_result
    global g_fd_results, n_prevTimeStamp
    global g_ssl_results
    global g_start_time

    g_start_time = get_current_time()

    bOnline = True
    #test_conversations = []
    #test_conversations.append("그래. 무슨일이지?")
    #test_labels = []
    #for id in range(len(test_conversations)):
    #    test_labels.append(0)   # forcing 0s because it's not important.
    #ml.test(test_conversations, test_labels, bOnline)

    n_prevTimeStamp = 0 # global
    g_speech_result = 0 # global
    g_ssl_results = []  # global
    g_speech_recognized = "NULL"    # global

    # Initialize for face detection
    # camera index 0: internal camera, 1~: external camera
    camera_idx = 0
    #camera_idx = 1
    fd_detector, cap, args = init_fd(camera_idx)
    #-------------------------------------



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
    server_thread_sst.daemon = True
    server_thread_sst.start()

    thread_gcs_stt = threading.Thread(target=speech_recog)
    thread_gcs_stt.daemon = True
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


    while True:
        try:

            #---------------------------
            #--- face detection
            img = face_detection(fd_detector, cap, args)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.imshow("Face Detection with SSD", img)
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
                g_ssl_results = []

                test_conversations = []
                test_conversations.append(g_speech_recognized)
            
                test_labels = []
                test_labels.append(0)
                #print("Google Cloud Speech thinks you said " + test_conversations[0])
                #print("Speaker Direction : {}".format(Mic_tuning.direction))

                #---------------------------
                #--- classification
                classification = ml.test(test_conversations, test_labels, bOnline)
                id = 0
                print(" #---- sc ----#")
                if classification[id] == 1:
                    sys.stdout.write(gcs_stt.RED)
                    sys.stdout.write("\033[K")
                    print(f"   [{classification[id] }]: SENIOR! ({g_speech_recognized})\n")
                elif classification[id] == 0:
                    sys.stdout.write(gcs_stt.BLUE)
                    sys.stdout.write("\033[K")
                    print(f"   [{classification[id]}]: JUNIOR! ({g_speech_recognized})\n")
                else:
                    print(f"   [{classification[id]}]: NOT DETERMINED! ({g_speech_recognized})\n")

                sys.stdout.write(gcs_stt.RESET)
                sys.stdout.write("\033[K")
                #---------------------------

                #---------------------------
                #--- sound source tracking (DoA)
                g_ssl_results = sst_check()
                #---------------------------

                print("Say something!")
                    

        except sr.UnknownValueError:
            print("Google Cloud Speech could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Cloud Speech service; {0}".format(e))


    cv2.destroyAllWindows()
    cap.stop()   
     
