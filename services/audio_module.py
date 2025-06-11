import asyncio
import threading
import logging
import edge_tts

logger = logging.getLogger(__name__)

class TTSProcessor:
    def __init__(self, loop: asyncio.AbstractEventLoop, voice="en-US-AriaNeural"):
        self.voice = voice
        self.stop_event = threading.Event()
        self.audio_queue = asyncio.Queue()
        self.loop = loop  # ✅ store the main event loop

    def stop(self):
        self.stop_event.set()

    def synthesize(self, text: str):
        self.stop_event.clear()
        thread = threading.Thread(target=self._run_synthesis, args=(text,), daemon=True)
        thread.start()

    def _run_synthesis(self, text: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._stream_tts(text))

    async def _stream_tts(self, text: str):
        
        try:
            communicate = edge_tts.Communicate(text=text, voice=self.voice)
            async for chunk in communicate.stream():
                if self.stop_event.is_set():
                    break
                if chunk["type"] == "audio":
                    
                    asyncio.run_coroutine_threadsafe(
                        self.audio_queue.put(chunk["data"]),
                        self.loop
                    )
        except Exception as e:
            print(f"[TTS] Error: {e}")
