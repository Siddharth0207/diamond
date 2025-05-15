import streamlit as st
from streamlit_chat import message
import uuid
from src.audio_utils import _record_until_silence_blocking
from src.whisper_utils import load_whisper_model, transcribe_audio
from src.langchain_utils import llm_text_response
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
min_silence_len = 2000
fs = 16000


# Async function to handle transcription
async def handle_transcription(audio_data, fs):
    """
    Handles the transcription of audio data using the Faster-Whisper model.

    This function loads the Whisper model asynchronously and transcribes the provided audio data.

    Args:
        audio_data (numpy.ndarray): The recorded audio data as a NumPy array.
        fs (int): The sampling rate of the audio data in Hz.

    Returns:
        str: The transcription result as a single string.
    """

    model = await load_whisper_model(size=model_size, device=device_type)
    response = await transcribe_audio(model, audio_data, fs)
    return response
# Main logic
if st.button("🎤 Start Listening"): # Display a button to start recording
    st.info("Listening... Speak now.")  # Inform the user that recording has started

    # Record audio until silence is detected or the maximum duration is reached
    audio_data, fs = _record_until_silence_blocking(
        fs=16000,   # Sampling rate in Hz
        max_duration=180,    # Maximum recording duration in seconds
        min_silence_len_ms=min_silence_len, # Minimum silence duration to stop recording
        silence_thresh_db=silence_thresh    # Silence threshold in dBFS
    )

    st.success("Recording complete. Transcribing...") # Notify the user that recording is complete


    # Run the async function using asyncio.run() to await transcribe_audio
    response = asyncio.run(handle_transcription(audio_data, fs))
    llm_response = asyncio.run(llm_text_response(response))

    # Display the transcription result in the Streamlit app
    message(response, is_user=True, key=f"User-{response}")
    message(llm_response, is_user=False , key=f"LLM-{llm_response}")
