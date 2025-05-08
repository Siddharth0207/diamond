import torch 
import os 
from TTS.api import TTS
import gradio as gr 

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name ='tts_models/en/ljspeech/fast_pitch').to(device)

def generate_audio(text='Hi how may I help you?'):
    os.makedirs('outputs', exist_ok=True)
    tts = TTS(model_name ='tts_models/en/ljspeech/fast_pitch').to(device)
    tts.tts_to_file(text=text, file_path='outputs/output.wav')
    return 'outputs/output.wav'

print(generate_audio())
