#---------------------
# ##### Error: pip install pyttsx3 pyobjc
#
# pip install py3-tts
# pip install --upgrade wheel
#
# https://www.linkedin.com/pulse/text-to-speech-using-pyttsx3-dhanushkumar-r
#---------------------

import pyttsx3

def text_to_speech(text, voice_id=None, output_file=None):
    engine = pyttsx3.init()
    
    # List available voices
    voices = engine.getProperty('voices')

 
    # If a specific voice is specified, use it; otherwise, use the default voice
    if voice_id is not None and 0 <= voice_id < len(voices):
        engine.setProperty('voice', voices[voice_id].id)

    voice_id = 0
    for voice in voices:
        if voice.languages==['ko_KR']:
            voice_id_kor = voice_id
            print(voice_id_kor)
            print(voice)
            engine.setProperty('voice', voices[voice_id].id)
        voice_id = voice_id + 1


    engine.setProperty('rate',200) # default: 200

    # Convert text to speech
    engine.say(text)
    
    # If an output file is specified, save the speech as an audio file
    if output_file:
        engine.save_to_file(text, output_file)
        engine.runAndWait()
    else:
        engine.runAndWait()

if __name__ == "__main__":
    text = "안녕하세요, 어제 잘 주무셨나요?"
    #text = "아, 그럼, 오늘은 좀 쉬셔야겠네요."
    #text = "This is a test program"
    output_file = "output_A.mp3"
    
    # Change the voice by specifying the voice_id (0 for the first voice, 1 for the second, and so on)
    voice_id = 0  # Change this to the desired voice
    
    text_to_speech(text, voice_id, output_file)