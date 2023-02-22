#!/usr/bin/env python

import speech_recognition as sr
r = sr.Recognizer()

with sr.Microphone() as source:
    print('듣고 있어요')
    audio = r.listen(source) # 마이크로부터 음성 듣기

try:
    #text = r.recognize_google(audio, language='en-US') #구글 API로 인식( 하루 50회로 제한된다고 함 )
    text = r.recognize_google(audio, language='ko') #구글 API로 인식( 하루 50회로 제한된다고 함 )
    print(text)

except sr.UnknownValueError:
    print('인식 실패') # 음성 인식 실패한 경우

except sr.RequestError as e:
    print('요청 실패:{0}'.format(e)) # API Key 오류 또는 네트워크 단절 등
