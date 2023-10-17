# ReSpeaker USB 4 Mic Array (ReSpeaker Mic Array v2.0)

See https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/

> The product is available at [Seeed](https://www.seeedstudio.com/ReSpeaker-Mic-Array-v2.0-p-3053.html)

![](http://respeaker.io/assets/images/usb_4_mic_array.png)

The ReSpeaker USB 4 Mic Array is the successor of the ReSpeaker USB 6+1 Mic Array. It has better built-in audio processing algorithms than the 6+1 Mic Array, so it has better audio recording quality, although it only has 4 microphones.

## Features
+ 4 microphones
+ 12 RGB LEDs
+ USB
+ built-in AEC, VAD, DOA, Beamforming and NS
+ 16000 sample rate

## How to prepare

### Install Drivers (Set only once on first use) --> No need to install ???
```
sudo ./driver/install.sh
sudo reboot
```
[Follow the drivers installation guide to see more details](https://github.com/WoongDemianPark/HRI/tree/main/MicArray/driver)

### Device Firmware Update (Set only once on first use)
The Microphone Array supports USB DFU. We have [a python script - dfu.py](https://github.com/WoongDemianPark/HRI/blob/main/MicArray/dfu.py) to do that.

```
pip install pyusb
python dfu.py --download 6_channels_firmware.bin        # with sudo if usb permission error
```

In OSX,
```
brew install libusb 
brew install cmake fftw
```

And export DYLD_LIBRARY_PATH as followos,
```
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
```

| firmware | channels | note |
|---------------------------------|----------|-----------------------------------------------------------------------------------------------|
| 1_channel_firmware.bin | 1 | processed audio for ASR |
| 6_channels_firmware.bin | 6 | channel 0: processed audio for ASR, channel 1-4: 4 microphones' raw data, channel 5: playback |

### Create virtual environment for the module.
```
conda create -n MA_conv python=3.9
conda activate MA_conv
```
  
### Install requirements.
In OSX,
```
brew install portaudio flac

pip install -r requirements.txt
```

Otherwise,
```
sudo apt install swig

sudo apt install portaudio19-dev

pip install -r requirements.txt
```
  
### If the 'usb' module is not installed correctly, install it manually.
```
sudo pip3 install pyusb
```

### Make ReSpeaker usb device to be accessible to not only root but a regular user.
Generate a rule to get the accessibility of ReSpeaker usb device
```
sudo vi /etc/udev/rules.d/ReSpeaker_usb_4_mic_array.rules
```
Type like
```
SUBSYSTEMS=="usb", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", OWNER="jschoi", MODE="0666"
```

### Install pocketsphinx-python 
```
git clone --recursive https://github.com/cmusphinx/pocketsphinx-python
cd pocketsphinx-python
pip install . # <-- python setup.py install
```

### respeaker_python_library
```
cd respeaker_python_library # git clone https://github.com/respeaker/respeaker_python_library.git
cd examples
```

### [Option] ODAS
Refer to https://github.com/respeaker/usb_4_mic_array
Refer to https://respeaker.io/4_mic_array 

ODAS (https://github.com/introlab/odas) is a very cool project to perform sound source localization, tracking, separation and post-filtering.

Get ODAS and build it
```
sudo apt install libfftw3-dev libconfig-dev libasound2-dev
git clone https://github.com/introlab/odas.git

```

Install ODAS Studio
```
git clone https://github.com/introlab/odas_web
cd odas_web
npm install
```

Start ODAS Studio
```
npm start
```

In 'ODAS Control' part (bottom-left), link 'ODAS Core' to 'odaslive' execution file and link 'ODAS Config' to 'odas.cfg' configuration file in the above 'odas' folder.
(Notice: you need to modify ip address in 'odas.cfg' file depending on your system)

Click 'Lunch ODAS' and modify 'Potential sources energy range' bar in the bottom-right as [0.15 1.0] 


### Audacity
Download Audacity program from https://www.audacityteam.org


### Google Cloud Speech-To-Text
Create Google Cloud account and download json key file.
Follow the blog guide as below link.
```
https://ehdrh789.tistory.com/29
```

Install Google Cloud SDK and requirements.
```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-402.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-402.0.0-linux-x86.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init

pip install --upgrade google-cloud-storage
pip install google-cloud-speech
conda install -c anaconda pyaudio
```

Initialize and authorize the Google Cloud Account using json key file.
Edit
```
export GOOGLE_APPLICATION_CREDENTIALS=/home/USERNAME/DOWNLOAD-FOLDER/gcloud-key-filename.json
./google-cloud-sdk/bin/gcloud auth activate-service-account --key-file="/home/USERNAME/DOWNLOAD-FOLDER/gcloud-key-filename.json"
```


## How to run

### Connect the ReSpeaker USB 4 Mic Array to the NUC PC using a USB cable.

To run
1. In one terminal, type
    ```
    python sr_doa.py 2>/dev/null  
    ```
    Here '2>/dev/null' is used to disable ALSA warning messages


2. In the other terminal, move to the folder where odaslive execution exists
     (~/work/sHRI_base/MicArray_conversion/odas/build/bin)
    Then, type
    ```
    ./odaslive -c ~/work/sHRI_base/MicArray_conversation/odas/odas.cfg 
    ```

### For individual test of ReSpeaker USB 4 Mic Array
run "doa.py' for Direction of Arrival.
```
sudo python3 doa.py
```

run "vad.py' for Voice Activity Detection.
```
sudo python3 vad.py
```
</br>

