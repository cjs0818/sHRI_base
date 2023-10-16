# conversation (Conversation Module)

It performs a binary classification between "junior (0)" and "senior (1)" using Korean BERT model given a dataset of "conversation" and "labels".

This source code have been implimented based on Ubuntu 22.04 (64bit)

Running environment is as follows:
```
CPU: Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
Python: 3.9
```

## How to prepare
### Install conda
In case of macOS, install Miniconda3 as 
```
https://beelinekim.tistory.com/103
```
 (https://github.com/jeffheaton/t81_558_deep_learning/blob/master/install/manual_setup.ipynb)


### Create Virtual Environment.
```
conda create -n conversation python=3.9
conda activate conversation
```

### Install requirements.
```
pip install -r requirements.txt
```

### Dataset
Dataset is required to have at least two lists - "conversations" and "labels". For instance,
```
conversations = [
    "형님",
    "어, 왔어? 안녕",
    "박사님",
    "그래. 무슨일이지?",
    "식사하셨나요?",
    "아님, 음료수라도 하실까요?" 
]
labels = [
    0,
    1,
    0,
    1,
    0,
    0
]
```

To get a large data for a training, youtube data can be used.
1. download a caption file (.sri) of youtube contents (UniConverter program has been used.)
2. Convert .sri to .csv by help of http://convert.4get.kr/ko/convert/csv) 
3. Modify head descriptions as "conversations" and "labels"
4. Create ./practice folder and prepare a dataset file "cjs.csv" from the above "3". The dataset must include two contents: "conversations" and "labels".

## How to run

### Train
To train the dataset of "conversations" and "labels", do as follows:
```
python 01_train.py
```
The trained models will be saved as "checkpoint-x.pt" (where x means the number of checkpoint) in ./practice/weights foloder.

### Test
To test a classification given conversation, prepare "model.pt" in ./practice/weights folder. (you can copy some checkpoint-x.pt to model.pt)
And type
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
