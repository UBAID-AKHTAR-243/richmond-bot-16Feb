#!/usr/bin/env python3
"""
chunk.py

Process drug and supplement manufacturing data from 'rich-data' folder (containing .txt files)
using a Parent-Child chunking strategy with SemanticChunker and paragraph-based fallback.
Outputs a structured JSON file with parent text, child text + embeddings, and metadata.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

# Updated LangChain imports (as of langchain v1.x)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
# For token counting
from transformers import AutoTokenizer

# Constants
CHILD_MIN_TOKENS = 128
CHILD_MAX_TOKENS = 256
PARENT_MIN_TOKENS = 512
PARENT_MAX_TOKENS = 1024
OVERLAP_TOKENS = 50
SEMANTIC_THRESHOLD = 0.75
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, local

# Separators that respect sentence boundaries
SENTENCE_SEPARATORS = ["\n\n", "\n", ".", "!", "?", ";", ":", " "]

def get_token_length(text: str, tokenizer) -> int:
    """Return the number of tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))

class DocumentProcessor:
    """
    Processes a single document file, applying parent-child chunking,
    metadata extraction, and embedding generation.
    """

    def __init__(self, embeddings_model: Embeddings, tokenizer):
        self.embeddings_model = embeddings_model
        self.tokenizer = tokenizer
        self.token_length_func = lambda text: get_token_length(text, tokenizer)

    def extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """
        Extract metadata from file name and content.
        Returns a dict with keys: source_file_name, title, page_number,
        drug_name, document_type, date.
        """
        metadata = {
            "source_file_name": file_path.name,
            "title": file_path.stem,  # default to filename without extension
            "page_number": None,       # not easily extractable; could be improved
            "drug_name": None,
            "document_type": "Manufacturing Data",  # default
            "date": None
        }

        # Try to get title from first non-empty line
        lines = content.strip().splitlines()
        if lines:
            first_line = lines[0].strip()
            if first_line:
                metadata["title"] = first_line

        # Try to extract drug name from filename or content
        # Simple heuristic: assume filename contains drug name
        stem = file_path.stem
        # Remove common prefixes/suffixes like "data", "info", etc.
        # For demonstration, just use the stem
        metadata["drug_name"] = stem

        # Try to extract a date (YYYY-MM-DD or similar) from content
        date_pattern = r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b'
        date_match = re.search(date_pattern, content)
        if date_match:
            metadata["date"] = date_match.group()

        return metadata

    def create_parent_chunks(self, text: str) -> List[str]:
        """
        Create parent chunks (512-1024 tokens) using SemanticChunker as primary,
        with fallback to paragraph-based RecursiveCharacterTextSplitter.
        Ensures chunk sizes are within limits, adjusting if necessary.
        """
        # Try SemanticChunker first
        try:
            semantic_splitter = SemanticChunker(
                embeddings=self.embeddings_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=SEMANTIC_THRESHOLD,
                sentence_split_regex=r'(?<=[.?!])\s+'  # split on sentence boundaries
            )
            parent_chunks = semantic_splitter.split_text(text)
        except Exception as e:
            # Fallback to paragraph-based splitter
            print(f"SemanticChunker failed, using paragraph fallback: {e}")
            fallback_splitter = RecursiveCharacterTextSplitter(
                separators=SENTENCE_SEPARATORS,
                chunk_size=PARENT_MAX_TOKENS,
                chunk_overlap=OVERLAP_TOKENS,
                length_function=self.token_length_func,
            )
            parent_chunks = fallback_splitter.split_text(text)

        # Post-process to ensure each chunk meets min/max token requirements
        adjusted_chunks = []
        for chunk in parent_chunks:
            token_count = self.token_length_func(chunk)
            if token_count < PARENT_MIN_TOKENS:
                # Too small: try to merge with previous chunk if possible
                if adjusted_chunks:
                    merged = adjusted_chunks[-1] + "\n\n" + chunk
                    if self.token_length_func(merged) <= PARENT_MAX_TOKENS:
                        adjusted_chunks[-1] = merged
                        continue
                # If merging not possible, keep as is (small chunk)
                adjusted_chunks.append(chunk)
            elif token_count > PARENT_MAX_TOKENS:
                # Too large: recursively split with smaller chunks
                sub_splitter = RecursiveCharacterTextSplitter(
                    separators=SENTENCE_SEPARATORS,
                    chunk_size=PARENT_MAX_TOKENS,
                    chunk_overlap=OVERLAP_TOKENS,
                    length_function=self.token_length_func,
                )
                sub_chunks = sub_splitter.split_text(chunk)
                adjusted_chunks.extend(sub_chunks)
            else:
                adjusted_chunks.append(chunk)

        return adjusted_chunks

    def create_child_chunks(self, parent_text: str) -> List[str]:
        """
        Create child chunks (128-256 tokens) from a parent chunk.
        Uses the same semantic-first approach but within parent context.
        """
        # Try SemanticChunker first
        try:
            semantic_splitter = SemanticChunker(
                embeddings=self.embeddings_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=SEMANTIC_THRESHOLD,
                sentence_split_regex=r'(?<=[.?!])\s+'
            )
            child_chunks = semantic_splitter.split_text(parent_text)
        except Exception as e:
            # Fallback to sentence-aware recursive splitter
            fallback_splitter = RecursiveCharacterTextSplitter(
                separators=SENTENCE_SEPARATORS,
                chunk_size=CHILD_MAX_TOKENS,
                chunk_overlap=OVERLAP_TOKENS,
                length_function=self.token_length_func,
            )
            child_chunks = fallback_splitter.split_text(parent_text)

        # Post-process to enforce token limits
        adjusted_chunks = []
        for chunk in child_chunks:
            token_count = self.token_length_func(chunk)
            if token_count < CHILD_MIN_TOKENS:
                # Too small: try to merge with previous
                if adjusted_chunks:
                    merged = adjusted_chunks[-1] + " " + chunk
                    if self.token_length_func(merged) <= CHILD_MAX_TOKENS:
                        adjusted_chunks[-1] = merged
                        continue
                adjusted_chunks.append(chunk)
            elif token_count > CHILD_MAX_TOKENS:
                # Too large: split further
                sub_splitter = RecursiveCharacterTextSplitter(
                    separators=SENTENCE_SEPARATORS,
                    chunk_size=CHILD_MAX_TOKENS,
                    chunk_overlap=OVERLAP_TOKENS,
                    length_function=self.token_length_func,
                )
                sub_chunks = sub_splitter.split_text(chunk)
                adjusted_chunks.extend(sub_chunks)
            else:
                adjusted_chunks.append(chunk)

        return adjusted_chunks

    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a single file: read content, extract metadata,
        create parent and child chunks, generate embeddings for children.
        Returns a list of parent records, each with its children.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            print(f"Warning: {file_path} is empty, skipping.")
            return []

        metadata = self.extract_metadata(file_path, content)

        # Create parent chunks
        parent_chunks = self.create_parent_chunks(content)

        result = []
        for parent_text in parent_chunks:
            # Create child chunks within this parent
            child_texts = self.create_child_chunks(parent_text)

            # Generate embeddings for each child
            child_records = []
            for child_text in child_texts:
                # Generate embedding (as list of floats)
                embedding = self.embeddings_model.embed_query(child_text)

                # Combine metadata with child-specific info (if any)
                child_metadata = metadata.copy()
                # Could add parent reference if needed
                child_records.append({
                    "text": child_text,
                    "embedding": embedding,
                    "metadata": child_metadata
                })

            result.append({
                "parent_text": parent_text,
                "child_chunks": child_records
            })

        return result

def main():
    # Check for required packages and import errors gracefully
    try:
        # Initialize embedding model and tokenizer
        print("Loading embedding model and tokenizer...")
        embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure you have installed the required packages:")
        print("  pip install langchain-text-splitters langchain-experimental langchain-community transformers sentence-transformers")
        return

    processor = DocumentProcessor(embeddings_model, tokenizer)

    # Find all .txt files in rich-data directory
    data_dir = Path("rich-data")
    if not data_dir.exists():
        raise FileNotFoundError("Directory 'rich-data' not found.")

    txt_files = list(data_dir.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in rich-data.")
        return

    all_chunks = []
    for file_path in txt_files:
        print(f"Processing {file_path}...")
        try:
            file_chunks = processor.process_file(file_path)
            all_chunks.extend(file_chunks)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Save to JSON
    output = {"chunks": all_chunks}
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done. Processed {len(txt_files)} files, created {len(all_chunks)} parent chunks.")

if __name__ == "__main__":
    main()