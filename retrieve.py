#!/usr/bin/env python3
"""
retrieve.py

Full retrieval pipeline for RAG chatbot:
1. HyDE (Hypothetical Document Embedding) generation
2. Hybrid search: BM25 (keyword) + ANN (dense vector)
3. Merge & deduplicate results (up to 50 unique chunks)
4. Cross-encoder re-ranking → top 5 chunks
5. Confidence check → fallback if too low
"""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from openai import OpenAI
import tiktoken

# -------------------- Configuration --------------------
# Paths
CHUNKS_JSON_PATH = os.getenv("CHUNKS_JSON_PATH", "chunks.json")
CHROMA_PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chunks")

# Models
HYDE_LLM_MODEL = os.getenv("HYDE_LLM_MODEL", "gpt-3.5-turbo")          # for HyDE generation
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")  # must match embeddings.py
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Search parameters
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "25"))
ANN_TOP_K = int(os.getenv("ANN_TOP_K", "25"))
HYBRID_WEIGHTS = (0.5, 0.5)   # (bm25_weight, ann_weight) – used only for hybrid score computation
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.1"))

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -------------------- Data Loading --------------------
def load_chunks_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load child chunks from chunks.json (output of chunk.py).
    Expected format: {"chunks": [{"parent_text": ..., "child_chunks": [{"text": ..., "embedding": ..., "metadata": ...}]}]}
    Returns a list of chunks, each with 'text', 'metadata', and a generated 'id'.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Chunks file not found: {json_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {json_path}: {e}")
        return []

    all_chunks = []
    for parent in data.get("chunks", []):
        for child in parent.get("child_chunks", []):
            text = child.get("text", "").strip()
            if not text:
                continue
            metadata = child.get("metadata", {})
            # Generate a unique ID using source and chunk_index if available, else fallback to hash
            source = metadata.get("source_file_name", "unknown")
            idx = metadata.get("chunk_index", 0)
            # Use a simple ID format; for deduplication we rely on text+metadata combination
            chunk_id = f"{source}_{idx}_{hash(text) % 10000}"
            all_chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": metadata
            })
    logger.info(f"Loaded {len(all_chunks)} chunks from {json_path}")
    return all_chunks

def init_chroma_collection():
    """Initialize ChromaDB client and get the collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Connected to ChromaDB collection '{COLLECTION_NAME}'")
        return collection
    except Exception as e:
        logger.error(f"Failed to get ChromaDB collection: {e}")
        return None

# -------------------- BM25 Index --------------------
def build_bm25_index(chunks: List[Dict[str, Any]]) -> Tuple[BM25Okapi, List[str]]:
    """
    Build BM25 index from chunk texts.
    Returns BM25 object and the list of tokenized texts (for reference).
    """
    # Simple tokenizer: lowercase and split on whitespace
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info(f"Built BM25 index with {len(tokenized_corpus)} documents")
    return bm25, tokenized_corpus

# -------------------- HyDE --------------------
def generate_hypothetical_document(query: str) -> str:
    """
    Use an LLM to generate a hypothetical document that would answer the query.
    """
    prompt = (
        "You are an AI assistant. Given a user query, generate a concise, factual passage "
        "that would serve as an ideal answer to the query. The passage should be self-contained "
        "and informative.\n\n"
        f"Query: {query}\n\n"
        "Hypothetical answer:"
    )
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=HYDE_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        hypo_doc = response.choices[0].message.content.strip()
        logger.debug(f"HyDE generated: {hypo_doc[:100]}...")
        return hypo_doc
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}. Falling back to original query.")
        return query

def embed_text(text: str) -> List[float]:
    """Generate OpenAI embedding for a given text."""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise

# -------------------- Normalization --------------------
def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.5] * len(scores)   # all equal
    return [(s - min_score) / (max_score - min_score) for s in scores]

# -------------------- Main Retrieval Pipeline --------------------
def retrieve(query: str) -> Dict[str, Any]:
    """
    Full retrieval pipeline:
    1. HyDE generation
    2. Hybrid BM25 + ANN search (top-50 candidates)
    3. Cross-encoder re-ranking (top-5)
    4. Fallback if confidence too low
    Returns dict with 'context' (list of top-5 chunks) or 'fallback' (str).
    """
    # ---------- Load data (cached in module or reload each time) ----------
    # For simplicity, we reload each call; in production you might cache.
    chunks = load_chunks_from_json(CHUNKS_JSON_PATH)
    if not chunks:
        return {"fallback": "No data available. Please check the system configuration."}

    # Build BM25 index (could also be built once and reused)
    bm25, tokenized_corpus = build_bm25_index(chunks)

    # ChromaDB collection
    collection = init_chroma_collection()
    if not collection:
        return {"fallback": "Vector database unavailable."}

    # ---------- HyDE ----------
    hypo_doc = generate_hypothetical_document(query)
    hypo_embedding = embed_text(hypo_doc)

    # ---------- ANN search (using HyDE embedding) ----------
    try:
        ann_results = collection.query(
            query_embeddings=[hypo_embedding],
            n_results=ANN_TOP_K,
            include=["documents", "metadatas", "distances"]
        )
        # ChromaDB returns lists for each field; we take the first (and only) query
        ann_docs = ann_results["documents"][0] if ann_results["documents"] else []
        ann_metadatas = ann_results["metadatas"][0] if ann_results["metadatas"] else []
        ann_distances = ann_results["distances"][0] if ann_results["distances"] else []
        # Convert distances to similarities (assuming cosine distance: similarity = 1 - distance)
        ann_similarities = [1 - d for d in ann_distances]
    except Exception as e:
        logger.error(f"ANN query failed: {e}")
        ann_docs, ann_metadatas, ann_similarities = [], [], []

    # Normalize ANN similarities across the retrieved set
    if ann_similarities:
        ann_similarities_norm = normalize_scores(ann_similarities)
    else:
        ann_similarities_norm = []

    # Build list of ANN result dicts (tentative, will merge)
    ann_chunks = []
    for doc, meta, sim_norm in zip(ann_docs, ann_metadatas, ann_similarities_norm):
        # Try to find a matching chunk in our loaded chunks to get a consistent ID
        # For simplicity, we'll generate an ID on the fly using metadata
        source = meta.get("source", "unknown")
        idx = meta.get("chunk_index", 0)
        chunk_id = f"{source}_{idx}_{hash(doc) % 10000}"
        ann_chunks.append({
            "id": chunk_id,
            "text": doc,
            "metadata": meta,
            "ann_score": sim_norm
        })

    # ---------- BM25 search ----------
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get top BM25_TOP_K indices
    top_bm25_indices = np.argsort(bm25_scores)[-BM25_TOP_K:][::-1]
    bm25_chunks = []
    for idx in top_bm25_indices:
        score = bm25_scores[idx]
        chunk = chunks[idx]
        bm25_chunks.append({
            "id": chunk["id"],
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "bm25_raw": score
        })

    # Normalize BM25 scores among the retrieved set
    if bm25_chunks:
        raw_scores = [c["bm25_raw"] for c in bm25_chunks]
        norm_scores = normalize_scores(raw_scores)
        for c, norm in zip(bm25_chunks, norm_scores):
            c["bm25_score"] = norm
    else:
        bm25_chunks = []

    # ---------- Merge & Deduplicate ----------
    # Use a dictionary keyed by chunk ID
    merged = {}
    for c in ann_chunks:
        merged[c["id"]] = {
            "id": c["id"],
            "text": c["text"],
            "metadata": c["metadata"],
            "ann_score": c["ann_score"],
            "bm25_score": None
        }
    for c in bm25_chunks:
        if c["id"] in merged:
            merged[c["id"]]["bm25_score"] = c["bm25_score"]
        else:
            merged[c["id"]] = {
                "id": c["id"],
                "text": c["text"],
                "metadata": c["metadata"],
                "ann_score": None,
                "bm25_score": c["bm25_score"]
            }

    # Convert to list and compute hybrid score (if both scores exist, else use the one that exists)
    merged_list = []
    for item in merged.values():
        b = item["bm25_score"]
        a = item["ann_score"]
        if b is not None and a is not None:
            hybrid = HYBRID_WEIGHTS[0] * b + HYBRID_WEIGHTS[1] * a
        elif b is not None:
            hybrid = b  # only BM25
        elif a is not None:
            hybrid = a  # only ANN
        else:
            continue  # should not happen
        item["hybrid_score"] = hybrid
        merged_list.append(item)

    # The merged pool size may be up to 50 (or less after dedup)
    logger.info(f"Merged pool has {len(merged_list)} unique chunks")

    # ---------- Cross-Encoder Re-Ranking ----------
    if not merged_list:
        return {"fallback": "No relevant documents found."}

    # Prepare pairs for cross-encoder: (query, doc_text)
    pairs = [(query, item["text"]) for item in merged_list]
    try:
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        cross_scores = cross_encoder.predict(pairs)
    except Exception as e:
        logger.error(f"Cross-encoder failed: {e}")
        return {"fallback": "Re-ranking model unavailable."}

    # Attach scores and sort
    for item, score in zip(merged_list, cross_scores):
        item["cross_score"] = float(score)

    top5 = sorted(merged_list, key=lambda x: x["cross_score"], reverse=True)[:5]

    # ---------- Confidence Check ----------
    if top5[0]["cross_score"] < CONFIDENCE_THRESHOLD:
        return {
            "fallback": "I'm sorry, I don't have enough information to answer your question accurately. Could you rephrase or ask something else?"
        }

    # ---------- Prepare Output ----------
    context = []
    for item in top5:
        context.append({
            "chunk_text": item["text"],
            "source": item["metadata"].get("source_file_name", "unknown"),
            "cross_encoder_score": item["cross_score"]
        })

    return {"context": context}

# -------------------- Standalone Test --------------------
if __name__ == "__main__":
    # Simple test
    test_query = "What are the side effects of ibuprofen?"
    result = retrieve(test_query)
    if "context" in result:
        print(f"Top {len(result['context'])} chunks:")
        for i, chunk in enumerate(result["context"], 1):
            print(f"{i}. Score: {chunk['cross_encoder_score']:.4f} | Source: {chunk['source']}")
            print(f"   Text: {chunk['chunk_text'][:200]}...\n")
    else:
        print("Fallback:", result["fallback"])