# src/embedder.py
import chromadb
from chromadb.utils import embedding_functions
import os

COLLECTION_NAME = "clinic_knowledge"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

# ── SINGLETON ────────────────────────────────────────────────────────────────
# This variable holds ONE shared client for the entire app session.
# "Singleton" means only one instance exists — everyone shares the same one.
_client = None
_collection = None


def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_collection():
    """
    Returns the SAME collection instance every time it's called.
    This prevents creating multiple empty clients on Streamlit Cloud.
    """
    global _client, _collection

    # If collection already exists — return it immediately
    if _collection is not None:
        return _collection

    # Detect environment
    is_cloud = not os.path.exists(
        os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    )

    if is_cloud:
        print("Using in-memory ChromaDB (cloud)")
        _client = chromadb.EphemeralClient()
    else:
        print("Using persistent ChromaDB (local)")
        _client = chromadb.PersistentClient(path=DB_PATH)

    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )

    return _collection


def ingest_chunks(chunks: list[dict]):
    """
    Stores chunks in ChromaDB.
    """
    collection = get_chroma_collection()
    existing_count = collection.count()

    if existing_count > 0:
        print(f"⚠️  Already has {existing_count} chunks. Skipping.")
        return

    print(f"📥 Ingesting {len(chunks)} chunks...")

    ids   = [chunk["id"]   for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]

    collection.add(ids=ids, documents=texts)
    print(f"✅ Stored {len(chunks)} chunks successfully!")