#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.
NOTE: This module requires the additional dependency `pyaudio`. To install
using pip:
    pip install pyaudio
Example usage:
    python transcribe_streaming_mic.py
"""

# [START speech_transcribe_streaming_mic]
from __future__ import division

import re
import sys

import threading, time
from diart import stream
from stt import stt_module

from google.cloud import speech
# from google.cloud.speech import enums
# from google.cloud.speech import types
# import pyaudio
from six.moves import queue

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


def script_stt(language_code):
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    #language_code = 'ko-KR'  # a BCP-47 language tag

    stt_module.main()

def main():
    # script_stt thread calling
    language_code = 'ko-KR'  # a BCP-47 language tag    
    stt_thread = threading.Thread(daemon=True, name="stt_thread", target=script_stt, args=(language_code,))
    stt_thread.start()

    stream.run()

    while(1):
        time.sleep(10)


if __name__ == '__main__':
    main()
# [END speech_transcribe_streaming_mic]