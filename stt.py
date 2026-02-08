import os
import torch
import tempfile
import langcodes
import logging
import asyncio
import threading
from fastapi import UploadFile
import whisper
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("STT_MODEL", "base")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazily-loaded Whisper model and a lock to serialize model initialization
_whisper_model = None
_model_lock = threading.Lock()

async def get_whisper_model():
    """Return the cached Whisper model, loading it in a thread if needed."""
    global _whisper_model
    if _whisper_model is None:
        with _model_lock:
            if _whisper_model is None:
                logger.info(f"Loading Whisper model '%s' on device %s", MODEL_NAME, DEVICE)
                # load_model is blocking and potentially expensive — run in a thread
                _whisper_model = await asyncio.to_thread(whisper.load_model, MODEL_NAME, device=DEVICE)
    return _whisper_model

async def transcribe(audio_file: UploadFile) -> dict:
    """
    Transcribe an uploaded audio file to text and detect language.
    """
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio_file.read())
            tmp_path = tmp.name

        # Get (or load) the model and run transcription off the event loop
        model = await get_whisper_model()
        result = await asyncio.to_thread(model.transcribe, tmp_path)

        # Detect language
        language_code = result.get("language", "unknown")
        try:
            language_name = langcodes.Language.get(language_code).language_name()
        except AttributeError:
            language_name = "Unknown"

        return {
            "text": result.get("text", "").strip(),
            "language_code": language_code,
            "language_name": language_name
        }

    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        return {
            "text": "",
            "language_code": "error",
            "language_name": "Error",
            "error": str(e)
        }

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
