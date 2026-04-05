# src/embedder.py
# Stage 3 + 4: Embeddings + ChromaDB
# Uses chromadb==0.4.24 which works on Streamlit Cloud

import os
import chromadb
from chromadb.utils import embedding_functions

COLLECTION_NAME = "clinic_knowledge"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def get_embedding_function():
    """
    Uses ChromaDB's built-in embedding function.
    Downloads all-MiniLM-L6-v2 model via onnxruntime.
    Same function MUST be used for both storing and searching.
    """
    return embedding_functions.DefaultEmbeddingFunction()


def _is_cloud() -> bool:
    """Detect if running on Streamlit Cloud."""
    return not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    )


def get_chroma_client():
    """
    Returns correct ChromaDB client based on environment.
    Cloud  → EphemeralClient (in-memory)
    Local  → PersistentClient (saves to disk)
    """
    if _is_cloud():
        print("🌐 Cloud detected — using EphemeralClient")
        return chromadb.EphemeralClient()
    else:
        print("💻 Local detected — using PersistentClient")
        return chromadb.PersistentClient(path=DB_PATH)


def get_chroma_collection(client):
    """
    Gets or creates the ChromaDB collection.
    Always pass the client explicitly — no globals!
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )


def ingest_chunks(chunks: list[dict], client) -> None:
    """
    Embeds and stores chunks in ChromaDB.
    Skips if already populated.
    """
    collection = get_chroma_collection(client)

    if collection.count() > 0:
        print(f"⚠️  Already has {collection.count()} chunks. Skipping.")
        return

    print(f"📥 Ingesting {len(chunks)} chunks...")
    ids   = [chunk["id"]   for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    collection.add(ids=ids, documents=texts)
    print(f"✅ Stored {len(chunks)} chunks!")