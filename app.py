import streamlit as st
import uuid
from audio_utils import _record_until_silence_blocking
from whisper_utils import load_whisper_model, transcribe_audio
from uuid import UUID, uuid4
import asyncio

st.set_page_config(page_title="🗣️ Real-Time STT", layout="centered")
st.title("🎙️ Real-Time Speech-to-Text with Faster-Whisper")

# Assign unique user_id per session
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

st.caption(f"🔑 Session ID: `{st.session_state['user_id']}`")

# Config
model_size = "base"
device_type = "cpu"
silence_thresh = -32
min_silence_len = 500
fs = 16000

# Load Whisper model

# Async function to handle transcription
async def handle_transcription(audio_data, fs):
    model = await load_whisper_model(size=model_size, device=device_type)
    response = await transcribe_audio(model, audio_data, fs)
    return response
# Main logic
if st.button("🎤 Start Listening"):
    st.info("Listening... Speak now.")
    
    audio_data, fs = _record_until_silence_blocking(
        fs=16000,
        max_duration=180,
        min_silence_len_ms=min_silence_len,
        silence_thresh_db=silence_thresh
    )

    st.success("Recording complete. Transcribing...")


    # Run the async function using asyncio.run() to await transcribe_audio
    response = asyncio.run(handle_transcription(audio_data, fs))
   
    st.markdown(f"**📝 Transcription:** `{response}`")
