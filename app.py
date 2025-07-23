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


@app.post("/query")
async def query_diamond(request: DiamondQueryRequest, fastapi_request: Request):
    session_id = fastapi_request.headers.get("x-session-id", "default_session")
    logging.info(f"/query endpoint called. Session: {session_id}, Query: {request.query}")
    finder = DiamondFinder("postgresql+asyncpg://postgres:0207@localhost:5432/postgres")
    try:
        result = await finder.find_diamonds(request.query)
        logging.info(f"Query successful for session {session_id}")
        return JSONResponse(content=result, headers={"x-session-id": session_id})
    except Exception as e:
        logging.error(f"Error in /query endpoint for session {session_id}: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)