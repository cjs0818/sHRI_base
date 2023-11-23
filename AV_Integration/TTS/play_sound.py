#---------------------
# pip install playsound PyObjC

import playsound
import time

tts_delay = 5

playsound.playsound("output_A_01_R.mp3")
time.sleep(tts_delay)
playsound.playsound("output_B_01_L.mp3")
time.sleep(tts_delay)
playsound.playsound("output_A_02_R.mp3")
