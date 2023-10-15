from tuning import Tuning
import usb.core
import usb.util
import time
 
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
#print dev
if dev:
    Mic_tuning = Tuning(dev)
    while True:
        if Mic_tuning.is_voice()==1:
            print(f"The detected sound is a voice from the degree of {Mic_tuning.direction}")
        elif Mic_tuning.is_voice()==0:
            print("Waiting for sound input")
        time.sleep(0.2)
