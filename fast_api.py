import tracemalloc  # Import tracemalloc to track memory allocations

# Start tracing memory allocations
tracemalloc.start()


from fastapi import FastAPI
from audio_utils import record_until_silence_async
from whisper_utils import load_whisper_model, transcribe_audio
from uuid import UUID, uuid4

app = FastAPI()

model = load_whisper_model(size="base", device="cpu")
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI backend!"}

@app.post("/start_recording/{user_id}")
async def start_recording(user_id: UUID):
    audio_data, fs = await record_until_silence_async(
        fs=16000,
        min_silence_len_ms=1000,
        silence_thresh_db=-32,
        user_id=user_id
    )
    model = load_whisper_model(size="base", device="cpu")
    transcription = await transcribe_audio(model, audio_data, fs)
    return {
        "user_id": user_id, 
        "transcription": transcription
        }
