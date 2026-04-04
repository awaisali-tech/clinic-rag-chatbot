# src/generator.py
# PURPOSE: Send retrieved context + user question to Groq LLM
# and get back a helpful, accurate answer.
# This is Stage 6 of our RAG pipeline — the "G" in RAG (Generation).

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq
from dotenv import load_dotenv
from src.retriever import retrieve_relevant_chunks, format_context_for_llm

# Load the GROQ_API_KEY from .env file
load_dotenv()


def get_groq_client():
    """
    Creates and returns a Groq client using the API key from .env
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found! "
            "Make sure it's set in your .env file."
        )

    return Groq(api_key=api_key)


# This is the instruction we give Groq BEFORE every conversation.
# A "system prompt" is like a job description for the AI —
# it tells Groq what role to play and what rules to follow.
SYSTEM_PROMPT = """You are a helpful and friendly assistant for a group of medical clinics.
Your job is to answer patient questions accurately using ONLY the information provided in the context below.

STRICT RULES you must always follow:
1. Only use information from the provided context. Never make up details.
2. If the context does not contain the answer, say exactly:
   "I'm sorry, I don't have that information. Please contact the clinic directly."
3. Always mention the clinic name when referring to specific services or doctors.
4. Keep answers concise, warm, and easy to understand.
5. Never give medical advice or diagnose conditions.
6. If asked something unrelated to the clinics, politely redirect the conversation.
"""


def generate_answer(user_question: str, chat_history: list = None) -> dict:
    """
    Full RAG pipeline for one user question:
      1. Retrieve relevant chunks from ChromaDB
      2. Format them as context
      3. Send to Groq with the system prompt
      4. Return the answer + the sources used

    Args:
        user_question : The patient's question as a string
        chat_history  : Optional list of previous messages for multi-turn chat
                        Format: [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        A dict with:
            - "answer"  : Groq's response text
            - "sources" : List of chunk IDs that were used as context
    """
    if chat_history is None:
        chat_history = []

    # ── STEP 1: Retrieve relevant chunks ────────────────────────────────
    retrieved_chunks = retrieve_relevant_chunks(user_question, n_results=3)
    context_text     = format_context_for_llm(retrieved_chunks)
    source_ids       = [chunk["id"] for chunk in retrieved_chunks]

    # ── STEP 2: Build the message list for Groq ──────────────────────────
    # Groq expects a list of messages like a conversation history.
    # We inject the retrieved context into the LATEST user message
    # so Groq always has fresh, relevant information to work with.

    # Build the user message with context attached
    user_message_with_context = f"""Please answer the following patient question.

CONTEXT (retrieved from clinic database):
{context_text}

PATIENT QUESTION:
{user_question}"""

    # Assemble full message list:
    # [system prompt] + [chat history so far] + [new user message]
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + chat_history
        + [{"role": "user", "content": user_message_with_context}]
    )

    # ── STEP 3: Call Groq API ────────────────────────────────────────────
    client   = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,   # Lower = more factual, less creative (good for medical)
        max_tokens=512     # Enough for a detailed answer, not too long
    )

    answer = response.choices[0].message.content

    return {
        "answer" : answer,
        "sources": source_ids
    }


# ── TEST BLOCK ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    test_questions = [
        "What are Dr. Ayesha Khan's working hours?",
        "Does the dental clinic offer emergency services?",
        "Can I get LASIK surgery and which doctor should I see?",
        "What is the weather like today?",   # Out-of-scope test
    ]

    print("=" * 55)
    print("   RAG CHATBOT — FULL PIPELINE TEST")
    print("=" * 55)

    for question in test_questions:
        print(f"\n👤 Patient  : {question}")
        print("-" * 55)

        result = generate_answer(question)

        print(f"🤖 Assistant: {result['answer']}")
        print(f"📚 Sources  : {', '.join(result['sources'])}")
        print()