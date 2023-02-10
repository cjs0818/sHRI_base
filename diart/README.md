# Real-time Speaker Diarization Module

## How to prepare

* Download or clone the "diart" module in HRI folder.

* Create virtual environment for the module.
  ```
  conda create -n diart python=3.9
  conda activate diart
  ```

* Install requirements.
  ```
  pip install -r requirements.txt
  ```

* Install CUDA.
  ```
  sudo apt-get update
  sudo apt-get -y install nvidia-cuda-toolkit
  ```

* Confirm the CUDA Version and Installation.
  ```
  nvcc -V
  ```

* Install PyTorch for the installed CUDA version.
  * https://pytorch.org/get-started/locally/#start-locally
  * CPUOnly version was used for the NUC PC.
  ```
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```
  
* Install PortAudio and Soundfile.
  ```
  conda install portaudio
  pip install PySoundFile
  conda install pysoundfile -c conda-forge
  ```

* Install Pyannote.audio 2.0.
  ```
  pip install pyannote.audio
  pip install git+https://github.com/pyannote/pyannote-audio.git@2.0.1#egg=pyannote-audio
  ```

* Install Diart Module.
  ```
  pip install diart
  ```

* Install the Hub library.
  ```
  conda install -c conda-forge huggingface_hub
  ```

* Create the account on Huggingface and get Access Tokens.
  * https://huggingface.co/

* Once you have your User Access Token, run the following command in your terminal.
  ```
  huggingface-cli login
  ```

## How to run

### Stream audio

#### From the command line

A recorded conversation:

```shell
diart.stream /path/to/audio.wav
```

A live conversation:

```shell
diart.stream microphone
```

See `diart.stream -h` for more options.

#### From python

Run a real-time speaker diarization pipeline over an audio stream with `RealTimeInference`:

```python
from diart.sources import MicrophoneAudioSource
from diart.inference import RealTimeInference
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig

config = PipelineConfig()  # Default parameters
pipeline = OnlineSpeakerDiarization(config)
audio_source = MicrophoneAudioSource(config.sample_rate)
inference = RealTimeInference("/output/path", do_plot=True)
inference(pipeline, audio_source)
```

For faster inference and evaluation on a dataset we recommend to use `Benchmark` instead (see our notes on [reproducibility](#reproducibility)).

## Powered by research

Diart is the official implementation of the paper *[Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation](/paper.pdf)* by [Juan Manuel Coria](https://juanmc2005.github.io/), [Hervé Bredin](https://herve.niderb.fr), [Sahar Ghannay](https://saharghannay.github.io/) and [Sophie Rosset](https://perso.limsi.fr/rosset/).


> We propose to address online speaker diarization as a combination of incremental clustering and local diarization applied to a rolling buffer updated every 500ms. Every single step of the proposed pipeline is designed to take full advantage of the strong ability of a recently proposed end-to-end overlap-aware segmentation to detect and separate overlapping speakers. In particular, we propose a modified version of the statistics pooling layer (initially introduced in the x-vector architecture) to give less weight to frames where the segmentation model predicts simultaneous speakers. Furthermore, we derive cannot-link constraints from the initial segmentation step to prevent two local speakers from being wrongfully merged during the incremental clustering step. Finally, we show how the latency of the proposed approach can be adjusted between 500ms and 5s to match the requirements of a particular use case, and we provide a systematic analysis of the influence of latency on the overall performance (on AMI, DIHARD and VoxConverse).

## Citation

If you found diart useful, please make sure to cite our paper:

```bibtex
@inproceedings{diart,  
  author={Coria, Juan M. and Bredin, Hervé and Ghannay, Sahar and Rosset, Sophie},  
  booktitle={2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},   
  title={Overlap-Aware Low-Latency Online Speaker Diarization Based on End-to-End Local Segmentation}, 
  year={2021},
  pages={1139-1146},
  doi={10.1109/ASRU51503.2021.9688044},
}
```

## License

```
MIT License

Copyright (c) 2021 Université Paris-Saclay
Copyright (c) 2021 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
