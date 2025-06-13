from RealtimeTTS import TextToAudioStream
from gtts import gTTS
import tempfile
import os
import sys
import threading
from pydub import AudioSegment

from pydub.playback import play

class GTTS_Engine:
    def __init__(self):
        self.queue = None
        self.timings = None
        self.engine_name = "gTTS"

    def synthesize(self, text, lang='en'):
        tts = gTTS(text=text, lang=lang, tld='co.in')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name

    def synthesize_to_pcm_chunks(self, text, chunk_size=4096):
        audio_file = self.synthesize(text)
        audio = AudioSegment.from_file(audio_file, format="mp3")
        pcm_data = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2).raw_data
        for i in range(0, len(pcm_data), chunk_size):
            yield pcm_data[i:i+chunk_size]
        self.cleanup(audio_file)
    def cleanup(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    def get_stream_info(self):
        return ('mp3', 1, 22050)

class GTTS_AudioStream(TextToAudioStream):
    def __init__(self, engine):
        super().__init__(engine)
        self.audio_files = []

    def feed(self, text):
        audio_file = self.engine.synthesize(text)
        self.audio_files.append(audio_file)
        return self.audio_files

    def synthesize(self, text, lang='en'):
        return self.engine.synthesize(text, lang=lang)

    def synthesize_to_pcm_chunks(self, text, chunk_size=4096):
        audio_file = self.synthesize(text)
        audio = AudioSegment.from_file(audio_file, format="mp3")
        pcm_data = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2).raw_data
        for i in range(0, len(pcm_data), chunk_size):
            yield pcm_data[i:i+chunk_size]
        self.engine.cleanup(audio_file)

    def play_async(self):
        def play_and_cleanup():
            for audio_file in self.audio_files:
                audio = AudioSegment.from_file(audio_file, format="mp3")
                play(audio)
                self.engine.cleanup(audio_file)
        threading.Thread(target=play_and_cleanup).start()

if __name__ == "__main__":
    engine = GTTS_Engine()
    stream = GTTS_AudioStream(engine)
    