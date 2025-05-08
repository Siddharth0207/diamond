import speech_recognition as sr
import pyaudio

r = sr.Recognizer()


with sr.Microphone() as source:
    print("Please Speak now...")
    audio = r.listen(source)
    speech_text = r.recognize_bing(audio)
    print(speech_text)

