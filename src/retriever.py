# src/retriever.py
# Stage 5: Retrieval — find most relevant chunks for a query

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.embedder import get_chroma_collection


def retrieve_relevant_chunks(query: str,
                              collection,
                              n_results: int = 3) -> list[dict]:
    """
    Searches ChromaDB for chunks most similar to the query.
    Accepts collection as parameter — no global state!

    Args:
        query      : User's question
        collection : ChromaDB collection object (passed from app.py)
        n_results  : How many chunks to return

    Returns:
        List of dicts with id, text, score
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    ids       = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    retrieved = []
    for i in range(len(ids)):
        retrieved.append({
            "id"   : ids[i],
            "text" : documents[i],
            "score": round(distances[i], 4)
        })

    return retrieved


def format_context_for_llm(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into clean string for LLM.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Source {i} — {chunk['id']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)