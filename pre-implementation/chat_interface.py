import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment, silence
import tempfile
from faster_whisper import WhisperModel
import time
import os 
from langchain_nvidia_ai_endpoints import ChatNVIDIA
# Or HuggingFace if preferred

# ---------- CONFIG ----------
model_size = 'tiny'
device = 'cpu'
silence_thresh = -40
min_silence_len_ms = 1500
fs = 16000

client = ChatNVIDIA(
  model="meta/llama-3.3-70b-instruct",
  api_key="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",
  task='chat',
  temperature=0.6,
  top_p=0.7,
  max_tokens=4096,
) # Or swap with HF pipeline

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Voice Chat", layout="wide")
st.title("🗣️ Voice Chat with Whisper + LLM")

# ---------- Load Whisper Model ----------
@st.cache_resource
def load_model(size, device_type):
    return WhisperModel(size, device=device_type, compute_type="int8" if device_type == "cpu" else "float16")

model = load_model(model_size, device)

# ---------- Record Until Silence ----------
def record_until_silence(fs=16000, max_duration=60):
    audio_buffer = []
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        audio_buffer.extend(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
        while time.time() - start_time < max_duration:
            time.sleep(0.5)
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav.write(temp_wav.name, fs, np.array(audio_buffer))
            audio_seg = AudioSegment.from_wav(temp_wav.name)
            silences = silence.detect_silence(audio_seg, min_silence_len=min_silence_len_ms, silence_thresh=silence_thresh)
            if silences:
                return np.array(audio_buffer), fs
    return np.array(audio_buffer), fs

# ---------- Get LLM Response ----------
def get_llm_response(prompt):
    try:
        response = client.invoke(prompt=prompt, max_tokens=4096, temperature=0.6, top_p=0.7)
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Error: {e}"



if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "You're a helpful assistant."}]

# --- Display chat ---
for i, msg in enumerate(st.session_state.chat_history[1:]):  # skip system
    if msg["role"] == "user":
        st.chat_message("user", avatar="🧑").markdown(msg["content"])
    else:
        st.chat_message("assistant", avatar="🤖").markdown(msg["content"])

# --- Trigger voice input ---
if st.button("🎙️ Speak"):
    st.info("Listening... start speaking.")
    audio_data, fs = record_until_silence()
    st.success("Processing...")

    # Save and transcribe
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, fs, audio_data)
    segments, _ = model.transcribe(temp_wav.name)
    user_text = " ".join([seg.text for seg in segments]).strip()

    if user_text:
        st.chat_message("user", avatar="🧑").markdown(user_text)
        st.session_state.chat_history.append({"role": "user", "content": user_text})

        # LLM response
        assistant_response = get_llm_response(st.session_state.chat_history)
        st.chat_message("assistant", avatar="🤖").markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
