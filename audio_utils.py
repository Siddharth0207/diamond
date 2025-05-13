import sounddevice as sd
import numpy as np
from io import BytesIO
import time
import asyncio
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment
from multiprocessing import Pool
import fastapi
import uvicorn
from concurrent.futures import ThreadPoolExecutor



executor = ThreadPoolExecutor()

def _record_until_silence_blocking(
    fs=16000, 
    max_duration=180,
    min_silence_len_ms=1000,
    silence_thresh_db=-30,

    device=None
    ):
    audio_buffer = []
    start_time = time.time()


    def callback(indata, frames, time_info, status):
        if status:
            print(f"[Stream Status] {status}")
        # Flatten the mono channel
        audio_buffer.extend(indata[:, 0].copy())

    with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback, device=device):
        while time.time() - start_time < max_duration:
            time.sleep(0.5)

            if len(audio_buffer) == 0:
                continue

            np_audio = np.array(audio_buffer, dtype='int16')

            # Get the number of samples that make up the last N ms
            tail_samples = int(fs * min_silence_len_ms / 1000)
            if len(np_audio) < tail_samples:
                continue

            # Get the last `min_silence_len_ms` portion
            np_tail = np_audio[-tail_samples:]

            # Write to BytesIO as valid WAV
            buffer = BytesIO()
            wav_write(buffer, fs, np_tail)
            buffer.seek(0)

            try:
                tail_seg = AudioSegment.from_file(buffer, format="wav")
            except Exception as e:
                print(f"[Error loading audio segment]: {e}")
                continue

            # Log dBFS for tuning
            end_slice = tail_seg
            #print(f"[DEBUG] dBFS: {end_slice.dBFS:.2f}")

            if end_slice.dBFS < silence_thresh_db:
                print("🔇 Detected silence. Ending recording.")
                return np_audio, fs

    return np.array(audio_buffer, dtype='int16'), fs


# Async wrapper for FastAPI use
async def record_until_silence_async(
    fs=16000,
    max_duration=180,
    min_silence_len_ms=1000,
    silence_thresh_db=-40,
    user_id=None
):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        _record_until_silence_blocking,
        fs,
        max_duration,
        min_silence_len_ms,
        silence_thresh_db
    )
