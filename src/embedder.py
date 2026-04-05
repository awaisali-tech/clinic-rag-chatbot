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
    Converts text to a numeric vector.
    Uses words + bigrams for better semantic matching.
    """
    text = text.lower()
    
    # Remove punctuation
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    
    words = text.split()
    vec = np.zeros(size)
    
    # Single words
    for word in words:
        idx = hash(word) % size
        vec[idx] += 1.0
    
    # Bigrams (pairs of consecutive words)
    # "root canal" → hash("root_canal") — captures phrases
    for i in range(len(words) - 1):
        bigram = words[i] + "_" + words[i+1]
        idx = hash(bigram) % size
        vec[idx] += 1.5  # bigrams weighted higher than single words
    
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def _expand_query(query: str) -> str:
    """
    Expands user query with related medical terms.
    Helps match informal patient language to clinical terms.
    """
    synonyms = {
        "toothache"    : "tooth pain dental dentist",
        "tooth pain"   : "dental dentist tooth",
        "eye problem"  : "eye ophthalmology vision optometry",
        "skin problem" : "dermatology skin dermatologist",
        "heart"        : "cardiology cardiac cardiologist",
        "child"        : "pediatrics pediatrician children",
        "kids"         : "pediatrics pediatrician children",
        "online"       : "telemedicine online consultation video",
        "virtual"      : "telemedicine online consultation",
        "teeth"        : "dental dentist orthodontics",
        "braces"       : "orthodontics dental teeth",
        "glasses"      : "optical prescription vision eye",
        "laser"        : "lasik laser eye surgery",
        "cleaning"     : "dental checkup teeth whitening",
    }
    
    expanded = query
    query_lower = query.lower()
    
    for keyword, expansion in synonyms.items():
        if keyword in query_lower:
            expanded = expanded + " " + expansion
    
    return expanded

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
    Finds most relevant chunks using cosine similarity + query expansion.
    """
    global _chunks_store, _embeddings_store

    if not _chunks_store:
        return []

    # Expand query with synonyms before searching
    expanded_query = _expand_query(query)
    query_vec = _text_to_vector(expanded_query)

    scores = []
    for i, emb in enumerate(_embeddings_store):
        score = cosine_similarity(query_vec, emb)
        scores.append((score, i))

    scores.sort(reverse=True)

    results = []
    for score, idx in scores[:n_results]:
        results.append({
            "id"   : _chunks_store[idx]["id"],
            "text" : _chunks_store[idx]["text"],
            "score": round(1 - score, 4)
        })

    return results


def get_chunk_count() -> int:
    return len(_chunks_store)