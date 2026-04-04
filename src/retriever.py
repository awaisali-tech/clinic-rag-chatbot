# src/retriever.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.embedder import search_chunks


def retrieve_relevant_chunks(query: str, n_results: int = 3) -> list[dict]:
    """
    Finds most relevant chunks for a user question.
    Now uses our lightweight in-memory search instead of ChromaDB.
    """
    return search_chunks(query, n_results=n_results)


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