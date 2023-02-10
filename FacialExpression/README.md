# Robot Facial Expression Module
Robot Facial Expression Module

This source code have been implimented based on Ubuntu 22.04 (64bit)

Running environment is as follows:
```
CPU: Intel(R) Core(TM) i7-1165G7 @ 2.80GHz
Python: 3.9
```

## How to prepare
Create Virtual Environment.
```
conda create -n FacialExpression python=3.9
conda activate FacialExpression
```

Download Facial Animation files and save it into './FacialExpression/animation/'
*IF THERE IS NO DIRECTORY, JUST MAKE IT'.
```
Download link: https://drive.google.com/file/d/1DbmihnOoa8JsQ0F-tsN3ih31zcnak_Iw/view?usp=share_link
```

## How to run
It is possible that an extra library needs to be installed to run this source code.
```
python main.py
```

## Citation
* For Facial Expression Module, the following paper is used.
Please cite [Park, Ung et al.,](https://ieeexplore.ieee.org/document/9515533) 
```BibTeX
@article{WoongDemianPark,
  title={Robot Facial Expression Framework for Enhancing Empathy in Human-Robot Interaction},
  author={Park, Ung and Minsoo, Kim and Youngeun, Jang and GiJae, Lee and KangGeon, Kim and Ig-Jae, Kim and Jongsuk, Choi},
  journal={10.1109/RO-MAN50785.2021.9515533},
  year={2021}
}
```
