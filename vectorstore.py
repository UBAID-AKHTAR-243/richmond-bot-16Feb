import os
import json
import logging
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_path: str, meta_path: str, embed_model_id: str, device: str):
        self.index_path = index_path
        self.meta_path = meta_path
        self.emb_model = SentenceTransformer(embed_model_id, device=device)
        self.dim = self.emb_model.get_sentence_embedding_dimension()

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            logger.info(f"FAISS index loaded from {index_path}")
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            logger.info("FAISS index initialized (new)")

        self.meta: List[Dict[str, Any]] = []
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self.meta.append(json.loads(line))
                    except Exception:
                        continue
            logger.info(f"Metadata loaded: {len(self.meta)}")

    def persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            for row in self.meta:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info("Vector store persisted")

    def add_texts(self, texts: List[str], sources: List[str]):
        vecs = self.emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        self.index.add(vecs)
        start_id = len(self.meta)
        for i, (t, s) in enumerate(zip(texts, sources)):
            self.meta.append({"id": start_id + i, "source": s, "text": t})
        self.persist()

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        q_vec = self.emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q_vec, k)
        results = []
        if idxs is not None and len(idxs) > 0:
            for score, idx in zip(scores[0], idxs[0]):
                if 0 <= idx < len(self.meta):
                    row = self.meta[idx]
                    results.append({"score": float(score), "source": row["source"], "text": row["text"]})
        return results

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks
