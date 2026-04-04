# src/embedder.py
# PURPOSE: Take our text chunks, embed them, and store in ChromaDB.
# This is Stage 3 + Stage 4 of our RAG pipeline combined.
# Run this logic ONCE via ingest.py — not on every chat message.

import chromadb
from chromadb.utils import embedding_functions
import os

# ── CONSTANTS ───────────────────────────────────────────────────────────────
# The name of our collection inside ChromaDB
# A "collection" is like a table in a regular database
COLLECTION_NAME = "clinic_knowledge"

# Path where ChromaDB saves its files on your computer
# os.path.dirname(__file__) = src/
# We go one level up to reach the project root, then into chroma_db/
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")


def get_embedding_function():
    """
    Returns ChromaDB's built-in embedding function.
    Uses the all-MiniLM-L6-v2 model via onnxruntime (no PyTorch needed).
    This SAME function must be used for both storing AND searching.
    """
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_collection():
    """
    Creates (or connects to existing) ChromaDB client and collection.
    Think of this as 'opening the database'.

    Returns:
        A ChromaDB collection object we can add to or query.
    """
    # PersistentClient saves data to disk (survives after program closes)
    # If chroma_db/ folder doesn't exist, ChromaDB creates it automatically
    client = chromadb.PersistentClient(path=DB_PATH)

    # get_or_create_collection:
    #   - First run  → creates a brand new empty collection
    #   - Later runs → connects to the existing one (no duplicate data)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"}  # cosine = best for text similarity
    )

    return collection


def ingest_chunks(chunks: list[dict]):
    """
    Takes our text chunks and stores them in ChromaDB.
    Embeddings are generated automatically by the embedding_function.

    Args:
        chunks: List of {"id": ..., "text": ...} dicts from chunker.py
    """
    collection = get_chroma_collection()

    # Check if data already exists — avoid duplicating on re-runs
    existing_count = collection.count()

    if existing_count > 0:
        print(f"⚠️  Collection already has {existing_count} chunks.")
        print("   Skipping ingestion to avoid duplicates.")
        print("   To re-ingest: delete the chroma_db/ folder and run again.")
        return

    print(f"📥 Ingesting {len(chunks)} chunks into ChromaDB...")

    # Separate IDs and texts into two parallel lists
    # ChromaDB's add() function needs them this way
    ids   = [chunk["id"]   for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]

    # Add everything to ChromaDB in one batch
    # ChromaDB automatically calls the embedding_function on each text
    collection.add(
        ids=ids,
        documents=texts   # "documents" is ChromaDB's word for text chunks
    )

    print(f"✅ Successfully stored {len(chunks)} chunks!")
    print(f"📁 Database saved at: {os.path.abspath(DB_PATH)}")