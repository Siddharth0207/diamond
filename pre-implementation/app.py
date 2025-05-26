from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import os
import sys 
import time 
import numpy as np
from uuid import UUID, uuid4
import requests
import json
from faster_whisper import WhisperModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware # Import for handling CORS
load_dotenv()
templates = Jinja2Templates(directory="templates")
app = FastAPI()

# Load the Whisper model globally when the app starts
# You can change the model size and device
model = WhisperModel("base", device="cpu", compute_type="int8")
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHUNK_DURATION_SEC = 2
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION_SEC)
CHUNK_BYTES = CHUNK_SAMPLES * SAMPLE_WIDTH




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allows all origins (for development purposes)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return ({"message": "this is Root"})

@app.get("/api/summary/{topic}")
async def get_summary(topic: str):  
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        task='chat',
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
    )
    
    gen_prompt = PromptTemplate(
        template="You are a very helpful asistant and you will create a brief summary about the {topic} in 8-9 lines",
        input_variables=["topic"]
    )
    
    final_prompt = PromptTemplate(
        template="You are assistant and you will create a 2-3 line summary about the {topic}", 
        input_variables=["topic"]
    )
    
    parser = StrOutputParser()

    llm_chain_1 = gen_prompt | llm | parser
    llm_chain_2 = final_prompt | llm | parser
    parallel_chain = llm_chain_1 | llm_chain_2
    response =  parallel_chain.ainvoke({"topic": topic})

    return ({"summary": response})

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    partial_buffer = bytearray()
    final_buffer = bytearray()
    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                chunk = data["bytes"]
                partial_buffer += chunk
                final_buffer += chunk

                if len(partial_buffer) >= CHUNK_BYTES:
                    to_process = partial_buffer[:CHUNK_BYTES]
                    partial_buffer = partial_buffer[CHUNK_BYTES:]

                    audio_array = np.frombuffer(to_process, dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = model.transcribe(audio_array, beam_size=5, language="en", vad_filter=True)
                    text = " ".join([s.text for s in segments])
                    await websocket.send_text(json.dumps({"type": "partial", "text": text}))

            elif "text" in data:
                control = json.loads(data["text"])
                if control.get("final"):
                    audio_array = np.frombuffer(final_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    segments, _ = model.transcribe(audio_array, beam_size=5, language="en", vad_filter=True)
                    full_text = " ".join([s.text for s in segments])
                    await websocket.send_text(json.dumps({"type": "final", "text": full_text}))
                    partial_buffer = bytearray()
                    final_buffer = bytearray()

            await asyncio.sleep(0.001)

    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
    finally:
        await websocket.close()
