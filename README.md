# Voice Agent for Sales and Chat

A Python-based voice agent designed for sales and chat applications. This project leverages FastAPI, WebSockets, and advanced audio processing to provide real-time voice interaction capabilities. It integrates with language models and audio transcription services to deliver intelligent conversational experiences.

## Features

- Real-time voice chat via WebSockets
- Audio transcription using Whisper
- Integration with language models (LangChain)
- Modular service and router architecture
- Ready for deployment with Uvicorn

## Project Structure

```text
app.py                # Main application entry point
routers/              # API routers (e.g., summary, websocket_audio)
services/             # Service modules (audio, langchain, whisper)
home.html             # Example HTML client
requirements.txt      # Python dependencies
setup.py, pyproject.toml # Packaging and metadata
```

## Installation

1. Clone the repository:

   ```powershell
   git clone <your-repo-url>
   cd diamond
   ```

2. (Optional) Create and activate a virtual environment:

   ```powershell
   python -m venv dia
   .\dia\Scripts\activate
   ```

3. Install dependencies:

   ```powershell
   uv pip install -r requirements.txt
   ```

   Or, to use uv for direct installation:

   ```powershell
   uv install
   ```

## Usage

Start the FastAPI server with Uvicorn:

```powershell
uvicorn app:app --reload
```

Open `home.html` in your browser for a sample client interface.

## Configuration

- Service and model configurations can be adjusted in the `services/` and `routers/` modules.
- For advanced settings, edit `pyproject.toml` or `setup.py` as needed.

## License

This project is licensed under the terms of the LICENSE file in this repository.

## Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [LangChain](https://github.com/hwchase17/langchain)

---

For more information, see the code and comments in each module.
To build this **real-time, multi-user, deployment-ready voice assistant** with VAD, STT, NLU, LLM, and TTS — the ideal approach is **modular, scalable, and fault-tolerant**, using microservices architecture.

Here’s a **step-by-step approach** broken into **phases**:

---

## ✅ Phase 0: **Prelim Setup**

### Tools and Infra

- **Backend:** FastAPI + WebSocket + Gunicorn + Uvicorn
- **Frontend:** HTML/JS (or React) + WebSocket client + Recorder.js + VAD
- **Infra:** Docker, Redis, and possibly Kubernetes (later)
- **Logging:** Loguru or Sentry
- **Monitoring:** Prometheus + Grafana

---

## ✅ Phase 1: **Core Voice Pipeline (Monolithic or Service-Stitched)**

### 🔹 Step 1: Real-Time Audio Capture with VAD

- Use **WebRTC or MediaRecorder API** + **WebSocket**
- Run **VAD** (Silero VAD or WebRTC VAD) on the client
- Send chunks (e.g., 1s or 2s) only when voice is detected

### 🔹 Step 2: FastAPI Backend — Real-Time Audio Ingestion

- Accept audio chunks over WebSocket
- Buffer and pass to STT
- Maintain **session ID (user ID)** for each connection

---

## ✅ Phase 2: **Speech-to-Text (STT)**

- Use **Faster-Whisper** or **Whisper.cpp**
- Maintain:

  - **Partial transcription** (every 2 seconds)
  - **Final transcription** (on VAD silence)

- Send partials back via WebSocket to update UI

---

## ✅ Phase 3: **NLU, Entity Recognition, and Session**

- Plug-in a simple pipeline:

  - `Text → spaCy / Transformers → Entities → JSON`

- Store session context in **Redis** keyed by `user_id`

---

## ✅ Phase 4: **Dialogue Manager**

- Accept transcription + extracted entities
- Decide:

  - If info is missing: Ask for update
  - If all info present: Proceed to LLM

- Manage dialogue turns using Redis

---

## ✅ Phase 5: **LLM Response Generation**

- Use:

  - Local LLM (e.g., LLaMA2 / Mistral)
  - OR Hosted API (e.g., OpenRouter/Groq)

- Use **LangChain** for context injection, RAG if needed
- Return curated final response

---

## ✅ Phase 6: **Text-to-Speech (TTS)**

- Use **Coqui TTS** in streaming mode
- Convert text response into audio chunks
- Stream audio back over **WebSocket** to the frontend

---

## ✅ Phase 7: **Frontend Chat-Like Interface**

- Show:

  - Live transcript as user speaks
  - Bot's response (text + audio)

- Maintain session chat history

---

## ✅ Phase 8: **Multi-User Support**

- Use:

  - Redis for per-session state
  - `user_id` in WebSocket headers or JWT tokens

- Spin up dedicated WebSocket handler per session

---

## ✅ Phase 9: **Deployment**

- **Containerize** each service:

  - STT, TTS, LLM, FastAPI Gateway

- Use:

  - **Nginx** as reverse proxy
  - **Docker Compose** or **K8s** for orchestration

- Use autoscaling based on:

  - STT queue size
  - LLM request latency
  - TTS throughput

---

## ✅ Phase 10: **Extras**

- RAG via **Chroma / Pinecone**
- Semantic Memory via **FAISS** + Redis
- User Authentication (optional): JWT / OAuth
- Metrics Dashboard: Prometheus + Grafana

---

## 📌 Example Request Flow

1. **User speaks**
2. → VAD triggers streaming
3. → Audio sent to FastAPI via WebSocket
4. → STT transcribes partial/final
5. → NLU + Entity extraction
6. → Dialogue Manager decides next step
7. → LLM generates response
8. → Coqui TTS streams audio response
9. → WebSocket sends audio back to user

---

## 🗂️ Project Folder Structure

```plaintext
voice-agent/
├── backend/
│   ├── app.py (FastAPI entry)
│   ├── websockets/
│   ├── services/
│   │   ├── stt_service.py
│   │   ├── tts_service.py
│   │   ├── llm_service.py
│   │   ├── nlu_service.py
│   │   └── dialogue_manager.py
│   ├── utils/
│   └── redis_store.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md
```

---

## Updates and Recent Changes

### Redis-Based Session Management

- Implemented a `RedisSessionManager` class (`redis_manager.py`) for robust, concurrent, per-session state management.
- All WebSocket sessions now use Redis for storing and retrieving user session data, enabling true multi-user support and isolation.

### WebSocket Audio Endpoint

- The `/ws/audio` WebSocket endpoint in the backend now:
  - Assigns a unique session ID to each connection.
  - Stores and updates conversation history in Redis for each session.
  - Keeps the WebSocket connection open for the entire session, only closing on error or disconnect.
  - Handles client disconnects and server shutdowns gracefully (suppresses noisy tracebacks).

### Frontend

- No changes required for Redis integration; the frontend continues to communicate via WebSocket as before.
- The frontend can be extended to pass a session ID for reconnects or persistent sessions if needed.

### Data Loading Utility

- Added `load_excel_to_pg.py` for loading Excel data into PostgreSQL using pandas and SQLAlchemy.
- Includes clear docstrings and comments for maintainability.

### Code Quality

- Added detailed docstrings and inline comments to major backend modules for clarity and maintainability.
- Improved error handling and session cleanup in WebSocket logic.

---
