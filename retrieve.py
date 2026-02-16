#!/usr/bin/env python3
"""
retrieve.py - Retrieve top 3 most relevant text chunks from ChromaDB for a given query.
Usage: python retrieve.py "your query here"
"""

import sys
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError  # Import the specific exception

def main():
    # Check command-line argument
    if len(sys.argv) < 2:
        print("Error: No query provided.")
        print("Usage: python retrieve.py \"your query\"")
        sys.exit(1)

    query = sys.argv[1]

    # Path to persistent ChromaDB
    DB_PATH = "./chroma_db"
    COLLECTION_NAME = "chunks"

    try:
        # Initialize persistent client
        client = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    except Exception as e:
        print(f"Error connecting to ChromaDB at '{DB_PATH}': {e}")
        sys.exit(1)

    # Check if collection exists – catch the correct exception
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except NotFoundError:
        print(f"Collection '{COLLECTION_NAME}' does not exist in the database.")
        print("Make sure you have ingested data into ChromaDB first.")
        sys.exit(1)

    # Check if collection is empty
    if collection.count() == 0:
        print(f"Collection '{COLLECTION_NAME}' is empty. No documents to search.")
        sys.exit(1)

    # Perform similarity search
    try:
        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents", "distances", "metadatas"]
        )
    except Exception as e:
        print(f"Error during query execution: {e}")
        sys.exit(1)

    # Extract results (query returns lists of lists, one per query text)
    documents = results['documents'][0] if results['documents'] else []
    distances = results['distances'][0] if results['distances'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []

    if not documents:
        print("No matching documents found.")
        sys.exit(0)

    # Print results
    print(f"\nTop {len(documents)} results for query: \"{query}\"\n")
    for i, (doc, dist, meta) in enumerate(zip(documents, distances, metadatas), start=1):
        print(f"Result {i}:")
        print(f"  Text    : {doc[:200]}..." if len(doc) > 200 else f"  Text    : {doc}")
        print(f"  Distance: {dist:.4f}")
        if meta:
            print(f"  Metadata: {meta}")
        print("-" * 50)

if __name__ == "__main__":
    main()