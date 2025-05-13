import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment, silence
import tempfile
from faster_whisper import WhisperModel
import time

import logging
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)


# --- Streamlit UI ---
st.title("🎙️ Real-Time Speech-to-Text (Faster-Whisper, Auto Stop)")

model_size = 'tiny'
device = 'cpu'
silence_thresh = -40
min_silence_len_ms = 2000

# --- Load Model ---
@st.cache_resource
def load_model(size, device_type):
    return WhisperModel(size, device=device_type, compute_type="int8" if device_type == "cpu" else "float16")

model = load_model(model_size, device)

# --- Recording & Detection Logic ---
def record_until_silence(fs=16000, max_duration=180):
    st.info("Recording... Start speaking.")
    audio_buffer = []
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        audio_buffer.extend(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
        while time.time() - start_time < max_duration:
            time.sleep(0.5)

            # Check if user has stopped speaking
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav.write(temp_wav.name, fs, np.array(audio_buffer))

            audio_seg = AudioSegment.from_wav(temp_wav.name)
            silences = silence.detect_silence(audio_seg, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh)
            if silences:
                return np.array(audio_buffer), fs

    return np.array(audio_buffer), fs  # Timeout fallback

# --- Main Logic ---
if st.button("🎤 Start Listening"):
    audio_data, fs = record_until_silence()
    st.success("Speech complete. Transcribing...")

    # Save and transcribe
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, fs, audio_data)

    segments, _ = model.transcribe(temp_wav.name)
    full_text = " ".join([seg.text for seg in segments])
    final_text_to_read = full_text.strip()
    st.markdown(f"**📝 Transcription:** `{final_text_to_read}`")
