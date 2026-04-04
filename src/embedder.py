# src/embedder.py
import chromadb
from chromadb.utils import embedding_functions
import os

COLLECTION_NAME = "clinic_knowledge"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_collection(client=None):
    """
    Creates or connects to ChromaDB collection.
    Accepts an optional client — if not provided, creates one.
    """
    if client is None:
        is_cloud = not os.path.exists(
            os.path.join(os.path.dirname(__file__), "..", "chroma_db")
        )
        if is_cloud:
            client = chromadb.EphemeralClient()
        else:
            client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def ingest_chunks(chunks: list[dict], client=None):
    """
    Stores chunks in ChromaDB.
    """
    collection = get_chroma_collection(client=client)
    existing_count = collection.count()

    if existing_count > 0:
        print(f"⚠️  Already has {existing_count} chunks. Skipping.")
        return collection

    print(f"📥 Ingesting {len(chunks)} chunks...")
    ids   = [chunk["id"]   for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    collection.add(ids=ids, documents=texts)
    print(f"✅ Stored {len(chunks)} chunks!")
    return collection