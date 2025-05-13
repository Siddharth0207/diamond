from faster_whisper import WhisperModel
import tempfile
import scipy.io.wavfile as wav
import numpy as np
import asyncio

async def load_whisper_model(size="tiny", device="cpu"):
    compute = "int8" if device == "cpu" else "float16"
    return WhisperModel(size, device=device, compute_type=compute)

async def transcribe_audio(model, audio_data, sample_rate=16000):
    # Save to temp WAV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        wav.write(temp_wav.name, sample_rate, np.array(audio_data, dtype='int16'))

        segments, _ =  model.transcribe(temp_wav.name)
        transcription_result = " ".join([seg.text for seg in segments])
        return  transcription_result
