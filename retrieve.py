#!/usr/bin/env python3
"""
retrieve.py - Module for retrieving relevant text chunks from ChromaDB.
Provides a function `retrieve_chunks(query, n_results=3)` returning documents,
distances, and metadatas. Can also be run as a standalone script.
"""

import sys
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

DB_PATH = "./chroma_db"
COLLECTION_NAME = "chunks"

def retrieve_chunks(query, n_results=3):
    """
    Retrieve top n_results chunks from ChromaDB for the given query.
    Returns a tuple: (documents, distances, metadatas)
    """
    try:
        client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    except Exception as e:
        raise RuntimeError(f"Error connecting to ChromaDB at '{DB_PATH}': {e}")

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' does not exist. Run ingestion first.")

    if collection.count() == 0:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' is empty. No documents to search.")

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "distances", "metadatas"]
        )
    except Exception as e:
        raise RuntimeError(f"Error during query execution: {e}")

    docs = results['documents'][0] if results['documents'] else []
    dists = results['distances'][0] if results['distances'] else []
    metas = results['metadatas'][0] if results['metadatas'] else []

    return docs, dists, metas

def main():
    """Standalone mode: accepts query from command line and prints results."""
    if len(sys.argv) < 2:
        print("Error: No query provided.")
        print("Usage: python retrieve.py \"your query\"")
        sys.exit(1)

    query = sys.argv[1]

    try:
        docs, dists, metas = retrieve_chunks(query)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not docs:
        print("No matching documents found.")
        sys.exit(0)

    print(f"\nTop {len(docs)} results for query: \"{query}\"\n")
    for i, (doc, dist, meta) in enumerate(zip(docs, dists, metas), start=1):
        preview = doc[:200] + "..." if len(doc) > 200 else doc
        print(f"Result {i}:")
        print(f"  Text    : {preview}")
        print(f"  Distance: {dist:.4f}")
        if meta:
            print(f"  Metadata: {meta}")
        print("-" * 50)

if __name__ == "__main__":
    main()