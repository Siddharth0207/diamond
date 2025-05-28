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

3. Install dependencies using uv:

   ```powershell
   uv pip install -r requirements.txt
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
