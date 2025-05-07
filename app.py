import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import numpy as np
from faster_whisper import WhisperModel

# Optional: For live transcription using NVIDIA Riva or Whisper
# from my_transcription_module import transcribe_audio

st.title("🎙️ Voice Chatbot with WebRTC")

recording = []
# State variable to toggle recording
if "recording" not in st.session_state:
    st.session_state["recording"] = False

# Initialize transcription in session state
if "transcription" not in st.session_state:
    st.session_state["transcription"] = ""

# Start/Stop Button
if st.button("🎤 Start Recording" if not st.session_state["recording"] else "⛔ Stop Recording"):
    st.session_state["recording"] = not st.session_state["recording"]


# WebRTC audio processing
class AudioReciever:
    def __init__(self,sample_rate=16000) -> None:
        self.buffer = []
        # Store the sample rate (adjust based on your audio stream)
        self.sample_rate = sample_rate
        # Initialize the faster-whisper model
        self.model = WhisperModel(
            model_size_or_path="small.en",  # Use 'tiny.en' for English, or 'base', 'small', etc.
            device="cpu",  # Use 'cuda' if GPU is available
            compute_type="int8"  # Optimize for speed
        )


    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        if st.session_state.get("recording"):
            pcm = frame.to_ndarray()
            self.buffer.append(pcm)
            self.process_transcription(pcm)
            # Stub: process pcm here
        return frame
    def process_transcription(self):
        # Transcribe the audio chunk
        segments, _ = self.model.transcribe(
            vad_filter=True,  # Voice activity detection to ignore silence
            language="en"  # Adjust based on your needs
        )

        # Collect transcription
        new_transcription = ""
        for segment in segments:
            new_transcription += segment.text + " "

        # Update session state with new transcription
        if new_transcription.strip():
            st.session_state["transcription"] += new_transcription
            # Optionally, trigger Streamlit to rerun and display the updated transcription
            st.rerun()
    return 



webrtc_streamer(
    key="basic-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    audio_processor_factory=AudioReciever,
    media_stream_constraints={"audio": True, "video": False},
)