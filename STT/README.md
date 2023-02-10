# Speech to Text Module
Speech to Text Module

This source code have been implimented based on Ubuntu 22.04 (64bit)

Running environment is as follows:
```
CPU: Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
Python: 3.9
```

## How to prepare
Create Virtual Environment.
```
conda create -n STT python=3.9
conda activate STT
```

Create Google Cloud account and download json key file.
Follow the blog guide as below link.
```
https://ehdrh789.tistory.com/29
```

Install Google Cloud SDK and requirements.
```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-402.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-402.0.0-linux-x86.tar.gz
<<<<<<< HEAD
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init
=======
./Downloads/google-cloud-sdk/install.sh
./Downloads/google-cloud-sdk/bin/gcloud init
>>>>>>> 2b13bf886d1fca371540d01166ac441bc884344f

pip install --upgrade google-cloud-storage
pip install google-cloud-speech
conda install -c anaconda pyaudio
```

Initialize and authorize the Google Cloud Account using json key file.
<<<<<<< HEAD
Edit 
```
set GOOGLE_APPLICATION_CREDENTIALS=/home/USERNAME/DOWNLOAD-FOLDER/gcloud-key-filename.json
export GOOGLE_APPLICATION_CREDENTIALS=/home/USERNAME/DOWNLOAD-FOLDER/gcloud-key-filename.json
./google-cloud-sdk/bin/gcloud auth activate-service-account --key-file="/home/USERNAME/DOWNLOAD-FOLDER/gcloud-key-filename.json"
=======
```
set GOOGLE_APPLICATION_CREDENTIALS=/home/USERNAME/DOWNLOAD-FOLDER/gcloud-key-filename.json
gcloud auth activate-service-account --key-file="/home/USERNAME/DOWNLOAD-FOLDER/gcloud-key-filename.json"
>>>>>>> 2b13bf886d1fca371540d01166ac441bc884344f
```

## How to run
It is possible that an extra library needs to be installed to run this source code.
```
<<<<<<< HEAD
python stt_test1.py
=======
python ./workspace/GoogleSTT/stt_test1.py
>>>>>>> 2b13bf886d1fca371540d01166ac441bc884344f
```

## Citation
* For Speech to Text Module, following content of the page is used.
Please cite [Google Cloud Speech-to-Text Documentation](https://cloud.google.com/speech-to-text/docs/speech-to-text-requests) 
```
The content of the guide is licensed under the Creative Commons Attribution 4.0 License,
and code samples are licensed under the Apache 2.0 License.
For details, see the Google Developers Site Policies.
```
