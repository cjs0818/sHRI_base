# conversation
conversation Module

It performs a binary classification between "junior (0)" and "senior (1)" using K-BERT transformers given dataset of "conversation" and "labels".

This source code have been implimented based on Ubuntu 22.04 (64bit)

Running environment is as follows:
```
CPU: Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
Python: 3.9
```

## How to prepare
Create Virtual Environment.
```
conda create -n conversation python=3.9
conda activate conversation
```

### Install requirements.
```
pip install -r requirements.txt
```


## How to run

### Dataset
Create ./practice folder and prepare a dataset file "cjs.csv".
The dataset must include two contents: "conversations" and "labels".

### Train
To train the dataset of "conversations" and "labels", do as follows:
```
python 01_train.py
```

### Test
To test a classification given conversation, do as follows:
```
python 02_test.py
```

## Citation
* For Speech to Text Module, following content of the page is used.
Please cite [Google Cloud Speech-to-Text Documentation](https://cloud.google.com/speech-to-text/docs/speech-to-text-requests) 
```
The content of the guide is licensed under the Creative Commons Attribution 4.0 License,
and code samples are licensed under the Apache 2.0 License.
For details, see the Google Developers Site Policies.
```
