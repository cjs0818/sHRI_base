#import STT
import speech_recognition as sr


def GetSpeechText():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("음성 인식 중...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio, language='ko-KR')
    except sr.UnknownValueError:
        print("음성 인식 불가")
    except sr.RequestError as e:
        print("Google Web Speech API 에러 발생: {0}".format(e))

if __name__ == '__main__':
	while True:
		try:
			#text = STT.GetSpeechText()
			text = GetSpeechText()
			print("음성 : " + text)
		except Exception as e:
			temp = 1