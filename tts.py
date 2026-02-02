"""
Text-to-Speech (TTS) module
---------------------------
Handles voice cloning and speech synthesis using Coqui XTTS v2.
Designed for FastAPI integration and AWS deployment.
"""

#import os
import torch
import logging
from pathlib import Path
from TTS.api import TTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Accept Coqui license automatically
Path("/root/.coqui").mkdir(parents=True, exist_ok=True)
Path("/root/.coqui/agreement_accepted.txt").write_text("accepted")

# Load XTTS v2 model once at startup
try:
    logger.info("📥 Loading XTTS v2 model...")
    tts_model = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
        gpu=(DEVICE == "cuda")
    )
    logger.info("✓ XTTS v2 model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load XTTS v2 model: {e}")
    raise


def synthesize(
    text: str,
    output_file: str,
    speaker_wav: str = "voice_sample.wav",
    language: str = "en"
) -> str:
    """
    Generate cloned speech from text using XTTS v2.

    Parameters
    ----------
    text : str
        Text to synthesize.
    output_file : str
        Path to save the generated audio file.
    speaker_wav : str
        Path to reference voice sample (WAV file).
    language : str
        Language code (e.g., "en", "ur").

    Returns
    -------
    str
        Path to the generated audio file.
    """
    try:
        logger.info(f"Synthesizing speech in {language}...")
        tts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_file,
            split_sentences=True
        )
        logger.info(f"✓ Audio saved at {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise
