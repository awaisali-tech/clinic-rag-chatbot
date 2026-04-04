import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq
from dotenv import load_dotenv
from src.retriever import retrieve_relevant_chunks, format_context_for_llm

load_dotenv()

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


def get_groq_client():
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found!")
    return Groq(api_key=api_key)


def generate_answer(user_question: str,
                    chat_history: list = None) -> dict:
    if chat_history is None:
        chat_history = []

    retrieved_chunks = retrieve_relevant_chunks(user_question, n_results=3)
    context_text     = format_context_for_llm(retrieved_chunks)
    source_ids       = [chunk["id"] for chunk in retrieved_chunks]

    user_message_with_context = f"""Please answer the following patient question.

CONTEXT (retrieved from clinic database):
{context_text}

PATIENT QUESTION:
{user_question}"""

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + chat_history
        + [{"role": "user", "content": user_message_with_context}]
    )

    client   = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )

    return {
        "answer" : response.choices[0].message.content,
        "sources": source_ids
    }