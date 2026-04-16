# GEMINI.md

## Project Overview

`voice-fastapi` is a high-performance speech recognition (ASR) and speaker identification backend system built with **FastAPI**. It is designed for real-time, low-latency audio processing, supporting both streaming and offline ASR modes.

### Key Features
- **Streaming ASR:** Real-time transcription using `sherpa-onnx` (Zipformer/Transducer models).
- **Offline ASR:** High-accuracy transcription using SenseVoice models.
- **Speaker Identification:** ONNX-based speaker embedding and matching (e.g., Eres2Net) to identify users by voice.
- **Voice Activity Detection (VAD):** Integrated energy-based and model-based VAD for efficient segment processing.
- **Audio Enhancement:** Pipeline for noise reduction and dereverberation.
- **Command Matching & Forwarding:** Fuzzy matching of transcribed text to predefined commands, with the ability to forward commands to external webhooks.
- **Persistent Storage:** SQLite (default) or other SQL databases via SQLAlchemy/SQLModel to store voiceprints, transcripts, and system settings.
- **WebSocket Interface:** Real-time bidirectional communication for audio streaming and metadata feedback.

### Main Technologies
- **Framework:** FastAPI, Uvicorn
- **Audio Processing:** Sherpa-ONNX, librosa, sounddevice, pyroomacoustics
- **Machine Learning:** ONNX Runtime
- **Database:** SQLAlchemy, SQLModel, SQLite
- **Utilities:** Pydantic, RapidFuzz, Jieba (Chinese tokenization)

---

## Building and Running

### Prerequisites
- Python 3.10+ (Conda recommended)
- Pre-trained models in `./models/` (ASR tokens/encoder/decoder, Speaker recognition ONNX).

### Setup
```bash
conda create -n voice-fastapi python=3.10 -y
conda activate voice-fastapi
pip install -r requirements.txt
```

### Running the Server
The server can be started using the provided shell scripts or directly via `main.py`.

**Directly:**
```bash
python main.py --config config/app_config.json
```

**Using Scripts:**
- `bash start.sh`: Starts the service (often used with systemd).
- `bash stop.sh`: Stops the service.

### Configuration
Configuration is managed through:
1.  **Command Line Arguments:** See `python main.py --help`.
2.  **JSON Config File:** Default is `config/app_config.json`.
3.  **Environment Variables:** (e.g., `DATABASE_URL`, `COMMAND_FORWARD_URL`).

---

## Project Structure & Architecture

### Core Components
- `app/api/`: REST and WebSocket endpoints.
    - `ws.py`: Entry point for real-time ASR sessions.
- `app/services/`: Business logic.
    - `voice/session.py`: Manages the lifecycle of an ASR session (`AsrSession`), coordinating VAD, ASR, and Speaker ID.
    - `voice/recognizer.py`: Wrapper for Sherpa-ONNX engines.
    - `voice/speaker.py`: Logic for computing voice embeddings and identifying users.
    - `voice/vad.py`: Voice Activity Detection implementations.
    - `audio_enhancement/`: Dereverb and spectral subtraction algorithms.
    - `command_forwarder.py`: Forwards matched commands to external services.
- `app/models.py` & `app/schemas.py`: Database models and Pydantic validation schemas.
- `app/database.py`: Database initialization and connection management.
- `config/`: Application configuration files.

### ASR Pipeline Flow
1.  **WebSocket Connection:** Client connects to `/ws/asr`.
2.  **Binary Audio:** Client streams raw PCM audio chunks.
3.  **VAD:** `AsrSession` uses VAD to detect start/end of speech.
4.  **Audio Enhancement:** (Optional) Audio is cleaned before processing.
5.  **ASR Engine:** Audio chunks are fed into Sherpa-ONNX for partial results.
6.  **Speaker ID:** Once enough audio is collected, the embedder identifies the speaker.
7.  **Finalization:** At the end of a speech segment, a final transcript is generated, commands are matched, and metadata is sent back to the client.

---

## Development Conventions

- **Coding Style:** Adheres to PEP 8. Uses `logging` for structured output (VAD transitions, ASR events).
- **Type Safety:** Extensive use of Python type hints and Pydantic for data validation.
- **Database Migrations:** The project uses manual "ensure" functions in `app/database.py` (e.g., `_ensure_user_columns`) to handle schema evolutions on startup.
- **Error Handling:** WebSocket sessions are wrapped in try-except blocks with event logging to the database.
- **Testing:** Test scripts are located in the `test/` directory, covering VAD, ASR, and Speaker Recognition.

---

## Important Commands
- **Swagger Docs:** `http://localhost:8000/docs`
- **Check Status:** `GET /status`
- **Lint/Format:** Use standard Python tools (e.g., `flake8`, `black`).
