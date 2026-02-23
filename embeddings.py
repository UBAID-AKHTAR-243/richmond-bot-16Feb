import json
import os
import logging
import hashlib
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI
import tiktoken

# -------------------- Configuration --------------------
# All settings can be overridden by environment variables
JSON_FILE = os.getenv("JSON_FILE", "chunks_nh.json")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chunks")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))          # texts per API call
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))         # model's max input tokens
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------- Token Helper --------------------
def truncate_text_to_token_limit(text: str, model: str = EMBEDDING_MODEL, limit: int = MAX_TOKENS) -> str:
    """Truncate text to fit within the model's token limit."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models (use cl100k_base which is common for recent models)
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > limit:
        logger.debug(f"Truncating text from {len(tokens)} to {limit} tokens")
        return enc.decode(tokens[:limit])
    return text

# -------------------- ID Generation --------------------
def generate_chunk_id(chunk: Dict[str, Any]) -> str:
    """Generate a deterministic unique ID for a chunk."""
    # Use source, chunk_index, and a hash of the text to detect changes
    source = chunk.get("source", "unknown")
    idx = chunk.get("chunk_index", 0)
    text = chunk.get("text", "")
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}_{idx}_{text_hash}"

# -------------------- Load Chunks (Generator) --------------------
def load_chunks(file_path: str):
    """Yield chunks one by one from a JSON array file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        raise

    if not isinstance(data, list):
        logger.error(f"Expected JSON array, got {type(data)}")
        raise TypeError("JSON file must contain an array of chunks")

    for idx, chunk in enumerate(data):
        if not isinstance(chunk, dict):
            logger.warning(f"Skipping item {idx}: not a dictionary")
            continue
        yield chunk

# -------------------- Main Pipeline --------------------
def main():
    logger.info("Starting embedding pipeline")

    # 1. Initialize OpenAI client (with built-in retries)
    client_openai = OpenAI(api_key=OPENAI_API_KEY, max_retries=3)

    # 2. Initialize ChromaDB client
    logger.info(f"Connecting to ChromaDB at {PERSIST_DIRECTORY}")
    client_chroma = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )

    # 3. Get or create collection and fetch existing IDs
    try:
        collection = client_chroma.get_collection(name=COLLECTION_NAME)
        logger.info(f"Collection '{COLLECTION_NAME}' exists")
        # Fetch all existing IDs (limit high enough to cover all; adjust if needed)
        existing = collection.get(limit=100000)  # Increase if you have more chunks
        existing_ids = set(existing["ids"])
        logger.info(f"Found {len(existing_ids)} existing chunks in collection")
    except Exception:
        collection = client_chroma.create_collection(name=COLLECTION_NAME)
        logger.info(f"Created new collection '{COLLECTION_NAME}'")
        existing_ids = set()

    # 4. Process chunks in batches
    chunks_processed = 0
    chunks_skipped = 0
    chunks_failed = 0

    batch_texts = []
    batch_metadatas = []
    batch_documents = []
    batch_ids = []

    # We'll iterate over chunks, collect a batch, then embed and insert
    for chunk in load_chunks(JSON_FILE):
        # Validate chunk
        text = chunk.get("text", "").strip()
        if not text:
            logger.warning(f"Skipping chunk with empty text: {chunk}")
            chunks_skipped += 1
            continue

        # Generate ID and check if already exists
        chunk_id = generate_chunk_id(chunk)
        if chunk_id in existing_ids:
            logger.debug(f"Skipping already embedded chunk: {chunk_id}")
            chunks_skipped += 1
            continue

        # Prepare metadata (keep it small)
        metadata = {
            "source": chunk.get("source", "unknown"),
            "chunk_index": chunk.get("chunk_index", 0),
            "word_count": chunk.get("word_count", 0)
        }

        # Truncate text if necessary
        safe_text = truncate_text_to_token_limit(text)

        # Add to current batch
        batch_ids.append(chunk_id)
        batch_metadatas.append(metadata)
        batch_documents.append(safe_text)
        batch_texts.append(safe_text)

        # If batch is full, process it
        if len(batch_texts) >= BATCH_SIZE:
            process_batch(
                client_openai, collection,
                batch_texts, batch_metadatas, batch_documents, batch_ids
            )
            chunks_processed += len(batch_texts)
            # Clear batch
            batch_texts = []
            batch_metadatas = []
            batch_documents = []
            batch_ids = []

    # Process any remaining chunks
    if batch_texts:
        process_batch(
            client_openai, collection,
            batch_texts, batch_metadatas, batch_documents, batch_ids
        )
        chunks_processed += len(batch_texts)

    logger.info(
        f"Pipeline finished. Processed: {chunks_processed}, "
        f"Skipped: {chunks_skipped}, Failed: {chunks_failed}"
    )

# -------------------- Batch Processing --------------------
def process_batch(
    client_openai,
    collection,
    texts: List[str],
    metadatas: List[Dict],
    documents: List[str],
    ids: List[str]
):
    """Generate embeddings for a batch and insert into ChromaDB."""
    if not texts:
        return

    logger.info(f"Processing batch of {len(texts)} chunks...")

    try:
        # 1. Generate embeddings via OpenAI
        response = client_openai.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        # Ensure order matches input
        embeddings = [item.embedding for item in response.data]

        # 2. Insert into ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        logger.info(f"Successfully added batch to ChromaDB")

    except Exception as e:
        logger.error(f"Batch failed: {e}", exc_info=True)
        # In production, you might write failed batch to a dead-letter queue
        # Here we just log and continue
        # Optionally, you could retry individually, but that's more complex
        # We'll mark them as failed for overall count
        global chunks_failed
        chunks_failed += len(texts)

if __name__ == "__main__":
    main()