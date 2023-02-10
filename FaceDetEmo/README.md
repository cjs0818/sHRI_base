# Real-time Face Detection and Emotion Recognition Module
Real-time Face Detection and Emotion Recognition Module

This souce code is highly encouraged from [`Deepface`](https://github.com/serengil/deepface).

For face detection, [`SSD`](https://sefiks.com/2020/08/25/deep-face-detection-with-opencv-in-python/) is used.
For facial expression recognition, [`Shi et al.,`](https://arxiv.org/abs/2103.10189) is used.

The execution speed (face detection + emotion recognition) is approx 0.06 sec per frame. To measure the execution speed, we basically assumed that only one face is detected. However, this source code also can work with multiple results of face detection.

This source code have been implimented based on Ubuntu 22.04 (64bit)


Running environment is as follows:
```
CPU: Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
Python: 3.9
```

## How to prepare
Install custom library of Deepface.
```
cd ./libs/deepface_mod
python setup.py install
```

Install Requirements.
```
conda install -c anaconda scikit-learn
pip install networks tqdm
```

Download FER (Facial Expression Recognition) model and save it into './models/RAF-DB/'
*IF THERE IS NO DIRECTORY, JUST MAKE IT'.
```
Download link: https://drive.google.com/file/d/1SLiSQpYMpjUEMYHDg_dBu3ffQMrKtigf/view?usp=share_link
```

Install PyTorch (Matching with CUDA version).
```
nvcc --version
https://pytorch.kr/get-started/locally/
conda install pytorch torchvision torchaudio cudatoolkit=11.5 -c pytorch -c conda-forge
```

## How to run
It is possible that an extra library needs to be installed to run this source code.
```
python main.py
```

## Citation

Please cite [deepface](https://ieeexplore.ieee.org/document/9259802) and [DDRL](https://ieeexplore.ieee.org/abstract/document/8451494) in your publications if it helps your research. Here is an example BibTeX entry:

```BibTeX
@inproceedings{serengil2020lightface,
  title={LightFace: A Hybrid Deep Face Recognition Framework},
  author={Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle={2020 Innovations in Intelligent Systems and Applications Conference (ASYU)},
  pages={23-27},
  year={2020},
  doi={10.1109/ASYU50717.2020.9259802},
  organization={IEEE}
}
```

```BibTeX
@inproceedings{yu2018deep,
  title={Deep discriminative representation learning for face verification and person re-identification on unconstrained condition},
  author={Yu, Jongmin and Ko, Donghwuy and Moon, Hangyul and Jeon, Moongu},
  booktitle={2018 25th IEEE International Conference on Image Processing (ICIP)},
  pages={1658--1662},
  year={2018},
  organization={IEEE}
}
```
* For Facial Expression Recognition a method of the following paper is used.
Please cite [Shi et al.,](https://arxiv.org/abs/2103.10189) 
```BibTeX
@article{shi2021learning,
  title={Learning to amend facial expression representation via de-albino and affinity},
  author={Shi, Jiawei and Zhu, Songhao},
  journal={arXiv preprint arXiv:2103.10189},
  year={2021}
}
```
