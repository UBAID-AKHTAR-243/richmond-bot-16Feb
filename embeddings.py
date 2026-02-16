import json
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

def main():
    # Configuration
    JSON_FILE = "chunks.json"
    COLLECTION_NAME = "chunks"
    PERSIST_DIRECTORY = "./chroma_db"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality

    # Load the chunk data
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: {JSON_FILE} not found. Please run chunk.py first.")
        return
    except json.JSONDecodeError:
        print(f"Error: {JSON_FILE} is not valid JSON.")
        return

    if not chunks:
        print("No chunks found in JSON file.")
        return

    # Initialize embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Initialize ChromaDB client (persistent)
    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY, settings=Settings(anonymized_telemetry=False))

    # Create or get collection
    # We'll use a simple function to get or create to avoid conflicts if collection exists
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists. Adding new chunks.")
    except Exception:
        collection = client.create_collection(name=COLLECTION_NAME)
        print(f"Created new collection '{COLLECTION_NAME}'.")

    # Prepare data for insertion
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for idx, chunk in enumerate(chunks):
        # Unique ID per chunk (source + chunk_index)
        source = chunk.get("source", "unknown")
        chunk_idx = chunk.get("chunk_index", 0)
        chunk_id = f"{source}_{chunk_idx}"

        # Text to embed
        text = chunk.get("text", "")
        if not text:
            print(f"Warning: chunk {idx} has no text, skipping.")
            continue

        # Metadata (store everything except text to keep metadata small)
        metadata = {
            "source": source,
            "chunk_index": chunk_idx,
            "word_count": chunk.get("word_count", 0)
        }

        # Generate embedding
        embedding = model.encode(text).tolist()

        ids.append(chunk_id)
        embeddings.append(embedding)
        metadatas.append(metadata)
        documents.append(text)

    # Add to ChromaDB in batches (recommended for large data)
    batch_size = 100
    print(f"Adding {len(ids)} chunks to ChromaDB...")
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
            documents=documents[i:end]
        )
        print(f"Added batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")

    print("Embeddings successfully stored in ChromaDB.")

if __name__ == "__main__":
    main()