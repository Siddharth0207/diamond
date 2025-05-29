# tts_engine.py
from TTS.api import TTS
import io
import numpy as np
import soundfile as sf

# Load model only once
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

def synthesize_streaming_chunks(text: str, sample_rate: int = 22050, chunk_size_ms: int = 250):
    """
    Synthesizes speech from text and yields small audio chunks (bytes) for streaming.

    Yields:
        PCM audio bytes suitable for WebSocket transmission.
    """
    # Generate entire waveform (NumPy array)
    wav = tts_model.tts(text)
    
    # Convert to PCM bytes
    pcm_bytes_io = io.BytesIO()
    sf.write(pcm_bytes_io, wav, samplerate=sample_rate, format='RAW', subtype='PCM_16')
    pcm_bytes = pcm_bytes_io.getvalue()

    # Calculate chunk size in bytes
    bytes_per_sample = 2  # 16-bit PCM
    chunk_bytes = int((sample_rate * chunk_size_ms / 1000) * bytes_per_sample)

    # Yield in chunks
    for i in range(0, len(pcm_bytes), int(chunk_bytes)):
        yield pcm_bytes[i:i + int(chunk_bytes)]
