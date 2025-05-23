import streamlit as st
import sounddevice as sd
import numpy as np
import asyncio
import websockets
import uuid

st.set_page_config(page_title="🗣️ Real-Time STT (WebSocket)", layout="centered")
st.title("🎙️ Real-Time Speech-to-Text with Faster-Whisper (WebSocket)")

# Assign unique user_id per session
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

st.caption(f"🔑 Session ID: `{st.session_state['user_id']}`")

fs = 16000
ws_url = f"ws://localhost:8000/ws/audio/{st.session_state['user_id']}"


# Async function to stream audio to WebSocket and handle transcription
async def stream_audio_to_ws():
    """
    Streams audio to the WebSocket server and handles the transcription and response display.

    This function connects to the WebSocket server, records audio in real-time, and sends the audio data to the server for transcription.
    It also receives the transcription and LLM response from the server and displays them in the Streamlit app.

    Args:
        None

    Returns:
        None
    """

    st.info("Listening... Speak now.")
    try:
        async with websockets.connect(ws_url, ping_interval=None) as ws:
            audio_buffer = []
            stop_flag = False
            # Get the main event loop here
            loop = asyncio.get_running_loop()

            silence_counter = 0
            silence_thresh = -32  # dBFS
            min_silence_chunks = int(2.0 / 0.1)  # 2 seconds of silence at 0.1s per chunk

            def callback(indata, frames, time_info, status):
                nonlocal silence_counter
                if status:
                    print(f"Sounddevice status: {status}")
                # Calculate dBFS for silence detection
                audio_np = indata.flatten().astype(np.float32)
                rms = np.sqrt(np.mean(audio_np ** 2))
                dbfs = 20 * np.log10(rms / 32768) if rms > 0 else -100
                if dbfs < silence_thresh:
                    silence_counter += 1
                else:
                    silence_counter = 0
                # Use the main event loop captured above
                asyncio.run_coroutine_threadsafe(ws.send(indata.tobytes()), loop)
                audio_buffer.extend(indata.copy())

            with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback, blocksize=int(fs * 0.1)):
                st.info("Recording... Speak now. Stop speaking to end.")
                while True:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                        if msg:
                            import json
                            data = json.loads(msg)
                            if data.get("type") == "partial":
                                st.write(f"Received {data['received']} bytes...")
                            elif data.get("type") == "final":
                                st.success(f"**Transcription:** {data['transcription']}")
                                st.info(f"**LLM Response:** {data['response']}")
                                return
                        # If enough silence detected, break and close connection
                        if silence_counter >= min_silence_chunks:
                            break
                    except asyncio.TimeoutError:
                        if silence_counter >= min_silence_chunks:
                            break
                        continue
            # After exiting the stream, close the websocket to trigger backend processing
            await ws.close()
            # Wait for the final message from backend
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
                if msg:
                    data = json.loads(msg)
                    if data.get("type") == "final":
                        st.success(f"**Transcription:** {data['transcription']}")
                        st.info(f"**LLM Response:** {data['response']}")
            except Exception:
                st.warning("No final response received from backend.")
    except Exception as e:
        st.error(f"WebSocket error: {e}")


# Main logic
if st.button("🎤 Start Real-Time Listening (WebSocket)"):
    asyncio.run(stream_audio_to_ws())
