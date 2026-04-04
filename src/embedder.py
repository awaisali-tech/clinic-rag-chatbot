# src/embedder.py
import chromadb
from chromadb.utils import embedding_functions
import os

COLLECTION_NAME = "clinic_knowledge"
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_collection():
    """
    Smartly chooses between:
    - PersistentClient (local computer) → saves to chroma_db/ folder
    - EphemeralClient  (Streamlit Cloud) → stores in memory
    
    Detects the environment automatically.
    """
    # Check if we're running on Streamlit Cloud
    # Streamlit Cloud sets this environment variable automatically
    is_cloud = os.environ.get("STREAMLIT_SHARING_MODE") or \
               os.environ.get("IS_STREAMLIT_CLOUD") or \
               not os.path.exists(os.path.join(
                   os.path.dirname(__file__), "..", "chroma_db"
               ))

    if is_cloud:
        # Cloud: use in-memory database (resets on each startup — that's ok)
        print("Using in-memory ChromaDB (cloud environment)")
        client = chromadb.EphemeralClient()
    else:
        # Local: use persistent database (survives restarts)
        print("Using persistent ChromaDB (local environment)")
        client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def ingest_chunks(chunks: list[dict]):
    """
    Stores chunks in ChromaDB.
    Works for both local and cloud environments.
    """
    collection = get_chroma_collection()
    existing_count = collection.count()

    if existing_count > 0:
        print(f"⚠️  Collection already has {existing_count} chunks. Skipping.")
        return

    print(f"📥 Ingesting {len(chunks)} chunks into ChromaDB...")

    ids   = [chunk["id"]   for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]

    collection.add(
        ids=ids,
        documents=texts
    )

    print(f"✅ Successfully stored {len(chunks)} chunks!")