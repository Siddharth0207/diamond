from fastapi import FastAPI
from pydantic import BaseModel 
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware # Import for handling CORS
from routers import websocket_audio , summary
from utils.logger import logging
from main import DiamondFinder
from fastapi.responses import JSONResponse
from fastapi import Request

load_dotenv()
templates = Jinja2Templates(directory="templates")
app = FastAPI(
    title="Voice Agent API",
    description="Real-time speech-to-text + LangChain + NVIDIA endpoints",
    version="1.0.0",
)


class DiamondQueryRequest(BaseModel):
    query: str




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allows all origins (for development purposes)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(summary.router, prefix="/api/v1/summary", tags=["Summary"])
app.include_router(websocket_audio.router, prefix="/api/v1/audio", tags=["WebSocket Audio"])

@app.get("/")
async def root():
    """
    Root endpoint for the FastAPI backend.

    Returns:
        dict: A welcome message indicating the API is running.
    """
    return ({"message": "this is Root"})
