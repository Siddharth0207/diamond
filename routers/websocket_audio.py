from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from faster_whisper import WhisperModel
from fastapi import APIRouter
from RealtimeTTS import TextToAudioStream

import json
import sys
import numpy as np
import asyncio
import base64
from uuid import uuid4


from .nlu_engine import DiamondEntityExtractor 

from services.tts_engine import GTTS_AudioStream , GTTS_Engine


from redis_manager import RedisSessionManager

router = APIRouter()
redis_manager = RedisSessionManager()

loop = asyncio.get_running_loop()  
#tts_processor = TTSProcessor(loop=loop) 

# Load the Whisper model globally when the app starts
# You can change the model size and device
model = WhisperModel("base", device="cpu", compute_type="int8")
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHUNK_DURATION_SEC = 2
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH


@router.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription using WhisperModel.
    Uses Redis for all session state to ensure safe, concurrent, multi-user support.

    Receives audio data in chunks from the client, buffers it, and processes it in segments.
    Performs speech-to-text transcription on each segment using the Whisper model and sends
    partial and final transcription results back to the client in real time.

    Args:
        websocket (WebSocket): The WebSocket connection instance.

    Protocol:
        - Receives binary audio data (PCM 16-bit, 16kHz) from the client.
        - Sends JSON messages with partial and final transcriptions.
        - Handles control messages (e.g., to indicate end of utterance).
    """
    await websocket.accept()  # Accept the WebSocket connection

    # === Assign session ID ===
    session_id = str(uuid4())
    # Initialize session in Redis with empty history
    redis_manager.set_session(session_id, {"history": []})
    partial_buffer = bytearray()  # Buffer for partial audio chunks
    final_buffer = bytearray()    # Buffer for the full utterance

    try:
        while True:
            try:
                data = await websocket.receive()  # Receive data from the client
            except Exception as e:
                print(f"[INFO] WebSocket receive error or disconnect: {e}")
                break

            if websocket.application_state == WebSocketState.DISCONNECTED:
                print(f"[INFO] Client disconnected: {session_id}")
                break

            if "bytes" in data:
                chunk = data["bytes"]  # Audio chunk from client
                partial_buffer += chunk  # Add to partial buffer
                final_buffer += chunk    # Add to final buffer

                # If enough audio is buffered, process it
                if len(partial_buffer) >= CHUNK_BYTES:
                    to_process = partial_buffer[:CHUNK_BYTES]
                    partial_buffer = partial_buffer[CHUNK_BYTES:]

                    # Convert bytes to float32 numpy array for Whisper
                    audio_array = np.frombuffer(to_process, dtype=np.int16).astype(np.float32) / 32768.0
                    # Transcribe the audio chunk (partial)
                    segments, _ = model.transcribe(audio_array, beam_size=5, language="en", vad_filter=True)
                    # Combine all segment texts
                    text = " ".join([s.text for s in segments])

                    # Send partial transcription to client
                    await websocket.send_text(json.dumps(
                        {"type": "partial",
                        "session_id": session_id,
                        "text": text})
                        )

            elif "text" in data:
                control = json.loads(data["text"])  # Control message from client

                if control.get("final"):
                    # If client signals end of utterance, process the full buffer
                    audio_array = np.frombuffer(final_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = model.transcribe(audio_array, beam_size=5, language="en", vad_filter=True)
                    full_text = " ".join([s.text for s in segments]).strip()
                    

                    if full_text:
                        # Update session history in Redis
                        session = redis_manager.get_session(session_id)
                        session.setdefault("history", []).append(full_text)
                        redis_manager.set_session(session_id, session)

                        # Prepare TTS message
                        tts_text =  full_text

                        # Send final transcription to client
                        # === Add this: Extract and send entities === #
                        #entities = extract_entities(full_text)
                        #await websocket.send_text(json.dumps(
                            #{"type": "entities",
                            #"session_id": session_id, 
                            #"data": entities})
                            #)
                        # === Generate TTS Response (Follow-up, etc.) ===
                        #followup_text = f"Got it. You said: {full_text}"
                       

                        await websocket.send_text(json.dumps({
                            "type": "final",
                            "session_id": session_id,
                            "text": full_text
                        }))
                        # Initialize TTS Processor
                        engine = GTTS_Engine()
                        tts_processor = GTTS_AudioStream(engine)
                        for chunk in tts_processor.synthesize_to_pcm_chunks(tts_text):
                            await websocket.send_bytes(chunk)
                        


                    # Reset buffers for next utterance
                    partial_buffer = bytearray()
                    final_buffer = bytearray()
                    

            await asyncio.sleep(0.001)  # Yield to event loop

    except WebSocketDisconnect:
        print(f"[INFO] WebSocket closed cleanly: {session_id}")

    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
        try:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except RuntimeError:
            pass

    finally:
        redis_manager.delete_session(session_id)
        try:
            if websocket.application_state != WebSocketState.DISCONNECTED:
                await websocket.close()
        except RuntimeError:
            pass
