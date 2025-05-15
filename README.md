try:
response = requests.post(f"http://localhost:8000/start_recording/{st.session_state['user_id']}")

        # Print raw response to debug
        st.code(response.text)
        st.code(response.status_code)


        response_data = response.json()
        st.success("Done!")
        st.markdown(f"**📝 Transcription:** `{response}`")
    except Exception as e:
        st.error(f"Error calling FastAPI: {e}")

# 🎙️ Real-Time Speech-to-Text with Faster-Whisper

This project is a real-time speech-to-text (STT) application built using the [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) library for transcription and [Streamlit](https://streamlit.io/) for the user interface. It records audio, detects silence to stop recording, and transcribes the audio into text using a Whisper model.

---

## 🚀 Features

- **Real-Time Audio Recording**: Records audio until silence is detected or a maximum duration is reached.
- **Speech-to-Text Transcription**: Uses Faster-Whisper for accurate and efficient transcription.
- **Streamlit UI**: Provides an interactive and user-friendly interface for recording and viewing transcriptions.
- **Customizable Parameters**: Allows configuration of model size, silence threshold, and more.

---

## 🛠️ Project Structure

GitHub Copilot
Here is a sample README.md file for your project:

voice_agent/
├── app.py # Main application file (Streamlit UI)
├── audio_utils.py # Handles audio recording and silence detection
├── whisper_utils.py # Loads Whisper model and handles transcription
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ⚙️ Configuration

The following parameters can be configured in `app.py`:

- **`model_size`**: The size of the Whisper model to use (e.g., `"tiny"`, `"base"`, `"large"`).
- **`device_type`**: The device to run the model on (`"cpu"` or `"cuda"` for GPU).
- **`silence_thresh`**: The silence threshold in dBFS to stop recording.
- **`min_silence_len`**: The minimum duration of silence (in milliseconds) to stop recording.
- **`fs`**: The sampling rate for audio recording (default: `16000` Hz).

---

## 🖥️ How It Works

1. **Start Recording**:

   - Click the "🎤 Start Listening" button in the Streamlit app.
   - The app listens for audio input and stops recording when silence is detected or the maximum duration is reached.

2. **Transcription**:
   - The recorded audio is passed to the Faster-Whisper model for transcription.
   - The transcription result is displayed in the Streamlit app.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/voice_agent.git
   cd voice_agent
   ```

2.Create a virtual environment and activate it:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Install additional system dependencies for audio processing:

   > FFmpeg: Required for handling audio files.
   > On Ubuntu: sudo apt install ffmpeg
   > On Windows: Download and install from FFmpeg.org.

▶️ Usage
Run the Streamlit app:

Open the app in your browser (default: http://localhost:8501).

Click the "🎤 Start Listening" button to start recording and transcribing audio.

📂 Key Files
app.py
The main application file that handles the Streamlit UI and integrates audio recording and transcription.
audio_utils.py
Contains functions for recording audio and detecting silence:
\_record_until_silence_blocking: Records audio until silence is detected or the maximum duration is reached.
whisper_utils.py
Contains functions for loading the Whisper model and transcribing audio:
load_whisper_model: Loads the Faster-Whisper model.
transcribe_audio: Transcribes audio data using the Whisper model.
🛠️ Customization
You can customize the following parameters in app.py:

Model Size: Change the model_size variable to use a different Whisper model size (e.g., "tiny", "base", "large").
Device: Set device_type to "cuda" for GPU acceleration (if available).
Silence Threshold: Adjust silence_thresh to fine-tune silence detection.
Minimum Silence Length: Modify min_silence_len to change how long silence must last to stop recording.
🧪 Example Output
1.Recording:
The app listens for audio input and stops when silence is detected. 2. Transcription:
The app displays the transcribed text in the UI.

📝 Transcription: "Hello, this is a test of the real-time speech-to-text application."

The transcribed text is displayed in the app:
🛡️ Requirements
Python 3.8 or higher
FFmpeg (for audio processing)
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

    🤝 Contributing
        Contributions are welcome! Feel free to open issues or submit pull requests.

    📧 Contact
        For questions or support, please contact siddharthkushwaha33@gmail.com.
