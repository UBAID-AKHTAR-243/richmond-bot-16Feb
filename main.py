"""
main.py - Production-ready FastAPI backend for RAG chatbot.

Integrates:
- retrieve.py: context retrieval
- augmenting_prompt.py: prompt building
- openai_llm.py: LLM response generation

Serves bot.html and provides REST endpoints.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import aiofiles
import openai
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
import uvicorn

# Import our modules
from retrieve import retrieve
from augmenting_prompt import get_default_builder
from openai_llm import LLMResponder

# -------------------- Configuration --------------------
class Settings:
    """Application settings loaded from environment variables."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")

    # Paths
    BASE_DIR: Path = Path(__file__).parent
    STATIC_DIR: Path = BASE_DIR / "static"
    AUDIO_DIR: Path = STATIC_DIR / "audio"

    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "500"))

    # Rate limiting
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "10/minute")

    # Audio cleanup (hours)
    AUDIO_MAX_AGE_HOURS: int = int(os.getenv("AUDIO_MAX_AGE_HOURS", "1"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

settings = Settings()

# -------------------- Logging --------------------
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Lifespan & Cleanup --------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up...")
    settings.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Start background task to clean old audio files
    cleanup_task = asyncio.create_task(cleanup_old_audio())

    yield

    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutdown complete.")

async def cleanup_old_audio():
    """Periodically delete audio files older than AUDIO_MAX_AGE_HOURS."""
    while True:
        try:
            cutoff = datetime.now() - timedelta(hours=settings.AUDIO_MAX_AGE_HOURS)
            for file in settings.AUDIO_DIR.glob("*.mp3"):
                mtime = datetime.fromtimestamp(file.stat().st_mtime)
                if mtime < cutoff:
                    file.unlink()
                    logger.debug(f"Deleted old audio: {file}")
        except Exception as e:
            logger.exception("Error during audio cleanup")
        await asyncio.sleep(3600)  # run every hour

# -------------------- Rate Limiting --------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG Chatbot", version="1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# -------------------- Middleware --------------------
# CORS – restrict to allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Host header validation
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# -------------------- Request ID Middleware --------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(f"Request {request_id} finished: {response.status_code}")
    return response

# -------------------- Static Files --------------------
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# -------------------- Global Components --------------------
# Async OpenAI client
aclient = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# PromptBuilder (default uses ./prompts folder)
builder = get_default_builder()

# LLMResponder (synchronous – will be run in thread pool)
responder = LLMResponder(
    builder=builder,
    model=settings.LLM_MODEL,
    temperature=settings.LLM_TEMPERATURE,
    max_tokens=settings.LLM_MAX_TOKENS,
    api_key=settings.OPENAI_API_KEY
)

# -------------------- Pydantic Models --------------------
class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User's question")

class ChatResponse(BaseModel):
    text: str
    audio_url: Optional[str] = None

class ErrorResponse(BaseModel):
    detail: str
    request_id: Optional[str] = None

# -------------------- Helper Functions --------------------
async def transcribe_audio(file_path: Path) -> str:
    """Transcribe audio file using OpenAI Whisper API (async)."""
    try:
        with open(file_path, "rb") as f:
            # Async transcription is not directly supported by openai library,
            # but we can run in thread pool.
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None,
                lambda: aclient.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            )
        return transcript.text
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail="Speech-to-text failed")

async def generate_tts(text: str) -> Path:
    """Generate TTS audio file using OpenAI TTS API (async)."""
    try:
        filename = f"{uuid.uuid4()}.mp3"
        file_path = settings.AUDIO_DIR / filename

        # Run synchronous API call in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: aclient.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
        )

        # Stream to file (synchronous, but we can write asynchronously)
        async with aiofiles.open(file_path, "wb") as f:
            # response.content is bytes
            await f.write(response.content)

        logger.info(f"TTS saved: {file_path}")
        return file_path
    except Exception as e:
        logger.exception("TTS failed")
        raise HTTPException(status_code=500, detail="Text-to-speech failed")

async def get_rag_response(question: str, request_id: str) -> dict:
    """
    Run full RAG pipeline (retrieval + LLM) in a thread pool to avoid blocking.
    """
    loop = asyncio.get_event_loop()

    # Run retrieval (synchronous) in thread pool
    retrieval_result = await loop.run_in_executor(None, retrieve, question)

    if "fallback" in retrieval_result:
        answer = retrieval_result["fallback"]
    else:
        context_docs = [
            {"text": item["chunk_text"]} for item in retrieval_result["context"]
        ]
        # Run LLM response in thread pool
        answer = await loop.run_in_executor(
            None,
            lambda: responder.respond(
                question=question,
                context_docs=context_docs,
                history=None
            )
        )

    # Generate TTS (optional – could be requested via query param)
    audio_url = None
    # For now, always generate TTS (you can make it conditional)
    audio_path = await generate_tts(answer)
    audio_url = f"/static/audio/{audio_path.name}"

    return {"text": answer, "audio_url": audio_url}

# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
@limiter.limit(settings.RATE_LIMIT)
async def get_index(request: Request):
    """Serve the main bot.html page."""
    html_path = settings.STATIC_DIR / "bot.html"
    if not html_path.exists():
        html_path = settings.BASE_DIR / "bot.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="bot.html not found")

    async with aiofiles.open(html_path, "r", encoding="utf-8") as f:
        html_content = await f.read()
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/chat", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT)
async def chat_endpoint(request: Request, chat_req: ChatRequest):
    """Handle text messages."""
    request_id = request.state.request_id
    question = chat_req.text.strip()
    logger.info(f"[{request_id}] Text question: {question[:50]}...")

    try:
        response_data = await get_rag_response(question, request_id)
        return ChatResponse(**response_data)
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/voice", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT)
async def voice_endpoint(
    request: Request,
    audio: UploadFile = File(...)
):
    """Handle voice messages."""
    request_id = request.state.request_id

    # Save uploaded file temporarily
    temp_filename = f"temp_{uuid.uuid4()}.webm"
    temp_path = settings.AUDIO_DIR / temp_filename
    try:
        contents = await audio.read()
        async with aiofiles.open(temp_path, "wb") as f:
            await f.write(contents)
        logger.info(f"[{request_id}] Saved uploaded audio: {temp_path}")

        # Transcribe
        question = await transcribe_audio(temp_path)
        logger.info(f"[{request_id}] Transcribed: {question[:50]}...")

        # Get RAG response
        response_data = await get_rag_response(question, request_id)
        return ChatResponse(**response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Unhandled error")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Clean up temp file
        if temp_path.exists():
            await asyncio.to_thread(temp_path.unlink)  # unlink may block, run in thread
            logger.debug(f"[{request_id}] Removed temp file: {temp_path}")

# -------------------- Run (for development only) --------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # Disable in production
        log_level=settings.LOG_LEVEL.lower()
    )