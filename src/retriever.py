# src/retriever.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.embedder import get_chroma_collection


def retrieve_relevant_chunks(query: str,
                              n_results: int = 3,
                              collection=None) -> list[dict]:
    """
    Finds most relevant chunks for a user question.
    Accepts optional collection — if not provided, creates one.
    """
    if collection is None:
        collection = get_chroma_collection()

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
    Formats retrieved chunks into clean context string for LLM.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i} — {chunk['id']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)