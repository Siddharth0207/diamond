import tracemalloc  # Import tracemalloc to track memory allocations

# Start tracing memory allocations
tracemalloc.start()


from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from src.audio_utils import record_until_silence_async
from src.whisper_utils import load_whisper_model, transcribe_audio
from src.langchain_utils import llm_text_response
from uuid import UUID, uuid4


app = FastAPI()

model = load_whisper_model(size="base", device="cpu")
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend!"}

@app.websocket("/ws/start_recording/{user_id}")
async def start_recording(websocket: WebSocket, user_id: UUID):
    await websocket.accept()
    audio_buffer = bytearray()
    audio_data, fs = await record_until_silence_async(
        fs=16000,
        min_silence_len_ms=1000,
        silence_thresh_db=-32,
        user_id=user_id
    )
    model = load_whisper_model(size="base", device="cpu")
    transcription = await transcribe_audio(model, audio_data, fs)
    response = await llm_text_response(transcription)
    return {
        "user_id": user_id, 
        "transcription": transcription,
        "response":response
        }

@app.websocket("/ws/audio/{user_id}")
async def websocket_audio(websocket: WebSocket, user_id: UUID):
    await websocket.accept()
    audio_buffer = bytearray()
    fs = 16000
    disconnected = False
    try:
        while True:
            # Receive audio chunk from client
            chunk = await websocket.receive_bytes()
            audio_buffer.extend(chunk)
            # Optionally: send partial status back
            await websocket.send_json({"type": "partial", "received": len(audio_buffer)})
    except WebSocketDisconnect:
        disconnected = True  # Just break out of the loop and process below
    # On disconnect, process the full audio
    if not disconnected and len(audio_buffer) > 0:
        import numpy as np
        # Convert bytes to numpy array (int16)
        np_audio = np.frombuffer(audio_buffer, dtype=np.int16)
        # Transcribe and respond
        model = await load_whisper_model(size="base", device="cpu")
        transcription = await transcribe_audio(model, np_audio, fs)
        response = await llm_text_response(transcription)
        await websocket.send_json({
            "type": "final",
            "user_id": str(user_id),
            "transcription": transcription,
            "response": response
        })
    
    
