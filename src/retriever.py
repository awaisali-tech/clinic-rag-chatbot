# src/retriever.py
# PURPOSE: Search ChromaDB for the most relevant chunks to a user's question.
# This is Stage 5 of our RAG pipeline — the "R" in RAG (Retrieval).
#
# How it works:
#   1. User's question → converted to embedding (same model as ingestion)
#   2. ChromaDB compares it against all 21 stored embeddings
#   3. Returns the top N most similar chunks (we use N=3)

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.embedder import get_chroma_collection


def retrieve_relevant_chunks(query: str, n_results: int = 3) -> list[dict]:
    """
    Finds the most relevant chunks for a given user question.

    Args:
        query     : The user's question as a plain string
        n_results : How many chunks to return (default: 3)

    Returns:
        A list of dicts, each with:
            - "id"   : the chunk ID  (e.g. "clinic_001_doctor_1")
            - "text" : the chunk text (the actual clinic information)
            - "score": similarity score (lower = more similar in cosine space)
    """
    # Get our ChromaDB collection (already has 21 chunks stored)
    collection = get_chroma_collection()

    # Query ChromaDB — it embeds the query and finds closest chunks
    # ChromaDB handles the embedding automatically using the same
    # embedding_function we assigned when creating the collection
    results = collection.query(
        query_texts=[query],   # list because ChromaDB supports batch queries
        n_results=n_results
    )

    # results looks like this (ChromaDB's raw format):
    # {
    #   "ids":       [["clinic_001_doctor_1", "clinic_001_overview", ...]],
    #   "documents": [["Clinic: Sunrise...",  "Clinic Name: ...",    ...]],
    #   "distances": [[0.21, 0.35, 0.48]]   ← lower = more similar
    # }
    # Notice the double brackets — ChromaDB wraps results in an extra list
    # because it supports querying multiple questions at once.
    # We only asked one question, so we take index [0] to unwrap.

    ids       = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    # Package results into clean, readable dicts
    retrieved = []
    for i in range(len(ids)):
        retrieved.append({
            "id"   : ids[i],
            "text" : documents[i],
            "score": round(distances[i], 4)   # round to 4 decimal places
        })

    return retrieved


def format_context_for_llm(chunks: list[dict]) -> str:
    """
    Joins retrieved chunks into one clean context string to send to the LLM.

    Instead of sending raw Python dicts to Gemini/Groq, we format them
    as readable numbered sections — easier for the LLM to parse.

    Args:
        chunks: The list returned by retrieve_relevant_chunks()

    Returns:
        A single formatted string containing all retrieved context
    """
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i} — {chunk['id']}]\n{chunk['text']}"
        )

    # Join all sources with a clear separator
    return "\n\n---\n\n".join(context_parts)


# ── TEST BLOCK ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    # Test queries — these simulate real patient questions
    test_queries = [
        "What are Dr. Ayesha's working hours?",
        "Does the dental clinic offer emergency services?",
        "What payment methods does Sunrise Medical accept?",
        "Can I get laser eye surgery?"
    ]

    print("=" * 55)
    print("   RETRIEVAL TEST — Searching ChromaDB")
    print("=" * 55)

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 55)

        # Get top 3 relevant chunks
        chunks = retrieve_relevant_chunks(query, n_results=3)

        for chunk in chunks:
            print(f"  📄 [{chunk['score']}] {chunk['id']}")
            # Show just the first line of the chunk text for brevity
            first_line = chunk['text'].split('\n')[0]
            print(f"       {first_line}")

        print()

    # Show one full formatted context so you see what the LLM will receive
    print("=" * 55)
    print("   FULL CONTEXT EXAMPLE (for LLM)")
    print("   Query: 'Tell me about Dr. Ayesha Khan'")
    print("=" * 55)
    chunks = retrieve_relevant_chunks("Tell me about Dr. Ayesha Khan")
    context = format_context_for_llm(chunks)
    print(context)