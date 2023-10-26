#---------------------
# pip install gTTS
#---------------------

from gtts import gTTS
import os

def text_to_speech(text, output_file):
    tts = gTTS(text, lang='ko')  # 'ko' is the language code for Korean
    tts.save(output_file)
    #os.system(f"start {output_file}")  # This command works on Windows

if __name__ == "__main__":
    text = "응. 잘 잤어 그런데 조금 피곤하긴 해."
    output_file = "output_B_01.mp3"
    text_to_speech(text, output_file)