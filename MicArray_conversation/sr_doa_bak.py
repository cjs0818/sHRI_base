#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import speech_recognition as sr

from tuning import Tuning
import usb.core
import usb.util
import time

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
Mic_tuning = Tuning(dev)
#print(Mic_tuning.direction)

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

# recognize speech using Sphinx
try:
    print("Sphinx thinks you said " + r.recognize_sphinx(audio))
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))

# recognize speech using Google Cloud Speech
#GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""INSERT THE CONTENTS OF THE GOOGLE CLOUD SPEECH JSON CREDENTIALS FILE HERE"""
GOOGLE_CLOUD_SPEECH_CREDENTIALS = "/home/jschoi/work/sHRI_base/STT/cjsstt.json"
try:
    print("Google Cloud Speech thinks you said " + r.recognize_google_cloud(audio, language="ko-KR", credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))
    print("Speaker Direction : {}".format(Mic_tuning.direction))

except sr.UnknownValueError:
    print("Google Cloud Speech could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Cloud Speech service; {0}".format(e))
