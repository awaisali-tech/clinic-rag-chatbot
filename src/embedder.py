# src/embedder.py
# Lightweight embedder using sentence-transformers via huggingface API
# No ChromaDB, no SQLite, no threading issues!

import os
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# We store everything in simple Python lists — no database needed!
# For 21 chunks, this is faster and more reliable than ChromaDB on cloud.

_chunks_store = []      # stores {"id": ..., "text": ...}
_embeddings_store = []  # stores numpy arrays (the number-lists)


def get_embedding(texts: list[str]) -> list:
    """
    Gets embeddings using Groq API is not available for embeddings,
    so we use a simple TF-IDF style approach with numpy.
    This is lightweight and works perfectly on Streamlit Cloud.
    """
    embeddings = []
    for text in texts:
        # Simple but effective: character-level bag of words
        # Convert text to a fixed-size vector using hashing
        vec = _text_to_vector(text)
        embeddings.append(vec)
    return embeddings


def _text_to_vector(text: str, size: int = 512) -> np.ndarray:
    """
    Converts text to a numeric vector using character hashing.
    Simple, fast, no external API needed.
    """
    text = text.lower()
    vec = np.zeros(size)
    words = text.split()
    for word in words:
        # Hash each word to a position in the vector
        idx = hash(word) % size
        vec[idx] += 1.0
    # Normalize the vector
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Measures how similar two vectors are.
    Returns a value between 0 (different) and 1 (identical).
    """
    return float(np.dot(vec1, vec2))


def ingest_chunks(chunks: list[dict]):
    """
    Stores chunks and their embeddings in memory.
    """
    global _chunks_store, _embeddings_store

    if len(_chunks_store) > 0:
        print(f"⚠️  Already ingested {len(_chunks_store)} chunks. Skipping.")
        return

    print(f"📥 Ingesting {len(chunks)} chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embedding(texts)

    _chunks_store = chunks
    _embeddings_store = embeddings
    print(f"✅ Ingested {len(chunks)} chunks successfully!")


def search_chunks(query: str, n_results: int = 3) -> list[dict]:
    """
    Finds most relevant chunks for a query using cosine similarity.
    This replaces ChromaDB's .query() method entirely.
    """
    global _chunks_store, _embeddings_store

    if not _chunks_store:
        return []

    query_vec = _text_to_vector(query)

    # Calculate similarity between query and every chunk
    scores = []
    for i, emb in enumerate(_embeddings_store):
        score = cosine_similarity(query_vec, emb)
        scores.append((score, i))

    # Sort by highest similarity first
    scores.sort(reverse=True)

    # Return top N results
    results = []
    for score, idx in scores[:n_results]:
        results.append({
            "id"   : _chunks_store[idx]["id"],
            "text" : _chunks_store[idx]["text"],
            "score": round(1 - score, 4)  # convert to distance format
        })

    return results


def get_chunk_count() -> int:
    return len(_chunks_store)