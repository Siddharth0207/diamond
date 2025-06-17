from RealtimeTTS import TextToAudioStream
from gtts import gTTS
import tempfile
import os
import sys
import threading
from pydub import AudioSegment

from pydub.playback import play

class GTTS_Engine:
    """
    Google Text-to-Speech (gTTS) Engine Wrapper for RealtimeTTS.

    This class provides methods to synthesize text to speech using gTTS, convert the result to PCM chunks for streaming,
    and clean up temporary files. It is designed to be used as a backend TTS engine for real-time applications.

    Args:
        None

    Attributes:
        queue (None): Placeholder for RealtimeTTS compatibility.
        timings (None): Placeholder for RealtimeTTS compatibility.
        engine_name (str): Name of the engine ('gTTS').

    Expected Behavior:
        - Synthesize text to MP3 using gTTS.
        - Convert MP3 to PCM for streaming.
        - Clean up temporary files after use.
    """
    def __init__(self):
        """
        Initialize the GTTS_Engine with required attributes for RealtimeTTS compatibility.
        """
        self.queue = None
        self.timings = None
        self.engine_name = "gTTS"

    def synthesize(self, text, lang='en'):
        """
        Synthesize text to speech using gTTS and save as a temporary MP3 file.

        Args:
            text (str): The text to synthesize.
            lang (str): Language/accent code (default 'en').

        Returns:
            str: Path to the generated MP3 file.
        """
        tts = gTTS(text=text, lang=lang, tld='co.in')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name

    def synthesize_to_pcm_chunks(self, text, chunk_size=4096):
        """
        Synthesize text to speech and yield raw PCM audio chunks for streaming.

        Args:
            text (str): The text to synthesize.
            chunk_size (int): Size of each PCM chunk in bytes (default 4096).

        Yields:
            bytes: PCM audio data chunk.

        Expected Behavior:
            - Converts MP3 to PCM (22050Hz, mono, 16-bit).
            - Yields PCM chunks for real-time streaming.
        """
        audio_file = self.synthesize(text)
        audio = AudioSegment.from_file(audio_file, format="mp3")
        pcm_data = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2).raw_data
        for i in range(0, len(pcm_data), chunk_size):
            yield pcm_data[i:i+chunk_size]
        self.cleanup(audio_file)

    def cleanup(self, filename):
        """
        Remove a temporary audio file if it exists.

        Args:
            filename (str): Path to the file to remove.

        Returns:
            None
        """
        if os.path.exists(filename):
            os.remove(filename)

    def get_stream_info(self):
        """
        Get the audio stream information for the engine output.

        Returns:
            tuple: (format, channels, sample_rate) for the audio stream.
        """
        return ('mp3', 1, 22050)

class GTTS_AudioStream(TextToAudioStream):
    """
    Audio Stream class for streaming TTS audio using GTTS_Engine.

    This class manages feeding text to the engine, streaming PCM chunks, and asynchronous playback.
    It is designed to work with RealtimeTTS and provide real-time audio streaming.

    Args:
        engine (GTTS_Engine): The TTS engine instance to use.

    Attributes:
        audio_files (list): List of generated audio file paths.

    Expected Behavior:
        - Feed text to the engine and store audio files.
        - Stream PCM audio chunks for real-time playback.
        - Support asynchronous local playback for testing.
    """
    def __init__(self, engine):
        """
        Initialize the audio stream with a GTTS_Engine instance.

        Args:
            engine (GTTS_Engine): The TTS engine instance.
        """
        super().__init__(engine)
        self.audio_files = []

    def feed(self, text):
        """
        Synthesize text and store the resulting audio file.

        Args:
            text (str): The text to synthesize.

        Returns:
            list: List of audio file paths generated so far.
        """
        audio_file = self.engine.synthesize(text)
        self.audio_files.append(audio_file)
        return self.audio_files

    def synthesize(self, text, lang='en'):
        """
        Synthesize text to speech using the engine.

        Args:
            text (str): The text to synthesize.
            lang (str): Language/accent code (default 'en').

        Returns:
            str: Path to the generated MP3 file.
        """
        return self.engine.synthesize(text, lang=lang)

    def synthesize_to_pcm_chunks(self, text, chunk_size=4096):
        """
        Synthesize text to speech and yield raw PCM audio chunks for streaming.

        Args:
            text (str): The text to synthesize.
            chunk_size (int): Size of each PCM chunk in bytes (default 4096).

        Yields:
            bytes: PCM audio data chunk.
        """
        audio_file = self.synthesize(text)
        audio = AudioSegment.from_file(audio_file, format="mp3")
        pcm_data = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2).raw_data
        for i in range(0, len(pcm_data), chunk_size):
            yield pcm_data[i:i+chunk_size]
        self.engine.cleanup(audio_file)

    def play_async(self):
        """
        Play all stored audio files asynchronously (for local testing).

        Returns:
            None
        """
        def play_and_cleanup():
            for audio_file in self.audio_files:
                audio = AudioSegment.from_file(audio_file, format="mp3")
                play(audio)
                self.engine.cleanup(audio_file)
        threading.Thread(target=play_and_cleanup).start()

if __name__ == "__main__":
    engine = GTTS_Engine()
    stream = GTTS_AudioStream(engine)
