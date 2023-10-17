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

# For ssl
import socket
import json
import threading
from threading import Lock
import math


# For stt
import speech_recognition as sr

from tuning import Tuning
import usb.core
import usb.util
import time

# For text_classification
from text_classification import MachineLearning

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



if __name__ == "__main__":

    # Replace "ODAS_SERVER_IP" and "ODAS_SERVER_PORT" with the desired IP and port for the server
    #odas_server_ip = "192.168.1.6"
    odas_server_ip = "192.168.1.21"
    odas_server_ssl_port = 9001
    odas_server_sst_port = 9000

    server_thread_ssl = threading.Thread(target=launch_socket_server, args=(odas_server_ip, odas_server_ssl_port))
    server_thread_ssl.start()

    server_thread_sst = threading.Thread(target=launch_socket_server, args=(odas_server_ip, odas_server_sst_port))
    server_thread_sst.start()

    #dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
    #Mic_tuning = Tuning(dev)
    #print(Mic_tuning.direction)

    # obtain audio from the microphone
    r = sr.Recognizer()


    # recognize speech using Google Cloud Speech
    #GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""INSERT THE CONTENTS OF THE GOOGLE CLOUD SPEECH JSON CREDENTIALS FILE HERE"""
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = "/home/jschoi/work/sHRI_base/STT/cjsstt.json"


    # text_classification
    #BASE_PATH='/home/jschoi/work/sHRI_base/conversation/'
    if sys.platform == "linux" or sys.platform == "linux2":
        BASE_PATH = '/home/jschoi/work/sHRI_base/conversation/' # for Linux
    elif sys.platform == "darwin":
        BASE_PATH='/Users/jschoi/work/sHRI_base/conversation/' # for macOS 
    ml = MachineLearning(BASE_PATH)

    bOnline = True
    #test_conversations = []
    #test_conversations.append("그래. 무슨일이지?")
    #test_labels = []
    #for id in range(len(test_conversations)):
    #    test_labels.append(0)   # forcing 0s because it's not important.

    #ml.test(test_conversations, test_labels, bOnline)

    n_prevTimeStamp = 0
    while True:
        try:
            with sr.Microphone() as source:
                #r.adjust_for_ambient_noise(source)
                print("Say something!")
                audio = r.listen(source)

            test_conversations = []

            # Google Cloud Speech-to-Text
            #test_conversations.append(r.recognize_google_cloud(audio, language="ko-KR", credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))

            # Google web speech API
            test_conversations.append(r.recognize_google(audio, language="ko-KR"))

            test_labels = []
            test_labels.append(0)
            print("Google Cloud Speech thinks you said " + test_conversations[0])
            #print("Speaker Direction : {}".format(Mic_tuning.direction))


            classification = ml.test(test_conversations, test_labels, bOnline)
            id = 0
            if classification[id] == 1:
                print(f" [{classification[id] }]: SENIOR! \n")
            elif classification[id] == 0:
                print(f" [{classification[id]}]: JUNIOR! \n")
            else:
                print(f" [{classification[id]}]: NOT DETERMINED! \n")


            #print(sst_az_list)
            data = sst_az_list
            
            if len(data) > 0 and n_prevTimeStamp != sst_az_stream['timeStamp']:
                print(f"n_prevTimeStamp: {n_prevTimeStamp}, current timeStamp: {sst_az_stream['timeStamp']}")
                n_prevTimeStamp = sst_az_stream['timeStamp']
                if 'activity' in data[0]:    # sst
                    if data[0]['activity'] > 0:
                        print("   ##### sst in stt #####")

                        for id in range(len(data)):
                            if data[id]['activity'] > 0:
                                print(f"   sst: {data[id]}")
                                data_x = data[id]['x']
                                data_y = data[id]['y']
                                azimuth = math.atan2(data_y, data_x) * 180 / math.pi
                                print("     azimuth: {:.1f} degree".format(azimuth))
                    

        except sr.UnknownValueError:
            print("Google Cloud Speech could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Cloud Speech service; {0}".format(e))
