# app.py
import streamlit as st
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))


def text_to_vector(text: str, size: int = 512) -> np.ndarray:
    text = text.lower()
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    words = text.split()
    vec = np.zeros(size)
    for word in words:
        idx = hash(word) % size
        vec[idx] += 1.0
    for i in range(len(words) - 1):
        bigram = words[i] + "_" + words[i+1]
        idx = hash(bigram) % size
        vec[idx] += 1.5
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def expand_query(query: str) -> str:
    synonyms = {
        "toothache"  : "tooth pain dental dentist",
        "tooth pain" : "dental dentist tooth",
        "root canal" : "dental endodontics teeth service",
        "eye problem": "eye ophthalmology vision optometry",
        "online"     : "telemedicine online consultation video",
        "virtual"    : "telemedicine online consultation",
        "teeth"      : "dental dentist orthodontics",
        "braces"     : "orthodontics dental teeth",
        "glasses"    : "optical prescription vision eye",
        "laser"      : "lasik laser eye surgery",
        "child"      : "pediatrics pediatrician children",
        "kids"       : "pediatrics pediatrician children",
    }
    expanded = query
    for keyword, expansion in synonyms.items():
        if keyword in query.lower():
            expanded = expanded + " " + expansion
    return expanded


def search_chunks(query: str,
                  chunks: list,
                  embeddings: list,
                  n_results: int = 3) -> list[dict]:
    if not chunks:
        return []

    expanded = expand_query(query)
    query_vec = text_to_vector(expanded)

    scores = []
    for i, emb in enumerate(embeddings):
        score = float(np.dot(query_vec, emb))
        scores.append((score, i))

    scores.sort(reverse=True)

    results = []
    for score, idx in scores[:n_results]:
        results.append({
            "id"   : chunks[idx]["id"],
            "text" : chunks[idx]["text"],
            "score": round(1 - score, 4)
        })
    return results


def format_context(chunks: list) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[Source {i} — {chunk['id']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# ── LOAD DATA ONCE ────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_data():
    """
    Loads clinic data and builds embeddings ONCE.
    Cached by Streamlit — survives reruns, shared across users.
    Returns chunks and embeddings directly — no global variables!
    """
    from src.data_loader import load_clinic_data
    from src.chunker import create_chunks

    data_path = os.path.join(
        os.path.dirname(__file__), "data", "clinic_data.json"
    )
    clinic_data = load_clinic_data(data_path)
    chunks      = create_chunks(clinic_data)

    # Build embeddings right here — no external module needed
    embeddings = [text_to_vector(chunk["text"]) for chunk in chunks]

    print(f"✅ Loaded {len(chunks)} chunks into Streamlit cache")
    return chunks, embeddings


# Load data — cached, shared across ALL users
chunks, embeddings = load_all_data()


def generate_answer(user_question: str,
                    chat_history: list = None) -> dict:
    """
    Full RAG pipeline — self contained in app.py
    No dependency on embedder.py module variables.
    """
    if chat_history is None:
        chat_history = []

    # Retrieve relevant chunks
    retrieved = search_chunks(user_question, chunks, embeddings, n_results=3)
    context   = format_context(retrieved)
    sources   = [r["id"] for r in retrieved]

    # Build prompt
    user_msg = f"""Please answer the following patient question.

CONTEXT (retrieved from clinic database):
{context}

PATIENT QUESTION:
{user_question}"""

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + chat_history
        + [{"role": "user", "content": user_msg}]
    )

    # Call Groq
    from groq import Groq
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")

    client   = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=512
    )

    return {
        "answer" : response.choices[0].message.content,
        "sources": sources
    }


SYSTEM_PROMPT = """You are a helpful and friendly assistant for a group of medical clinics.
Answer patient questions using ONLY the information in the context provided.

STRICT RULES:
1. Only use information from the provided context. Never make up details.
2. If the context does not contain the answer, say:
   "I'm sorry, I don't have that information. Please contact the clinic directly."
3. Always mention the clinic name when referring to services or doctors.
4. Keep answers concise, warm, and easy to understand.
5. Never give medical advice or diagnose conditions.
6. If asked something unrelated to clinics, politely redirect.
"""


# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinic Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Clinic Assistant")
st.caption(
    "Ask me anything about Sunrise Medical Center, "
    "Green Leaf Dental, or Wellness Eye Clinic."
)

# DEBUG — remove after confirming it works
st.sidebar.write(f"✅ Chunks loaded: {len(chunks)}")

st.divider()


# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your clinic assistant. I can help you with:\n\n"
                "- 🏥 **Sunrise Medical Center** — General, Pediatrics, Cardiology\n"
                "- 🦷 **Green Leaf Dental** — Dental & Oral Care\n"
                "- 👁️ **Wellness Eye Clinic** — Eye Care & Surgery\n\n"
                "What would you like to know?"
            )
        }
    ]


# ── DISPLAY CHAT HISTORY ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── HANDLE USER INPUT ─────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about doctors, services, timings...")

if user_input:

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    groq_history = []
    for msg in st.session_state.messages[1:-1]:
        if msg["role"] in ["user", "assistant"]:
            groq_history.append({
                "role"   : msg["role"],
                "content": msg["content"]
            })

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching clinic records... please wait"):
            result = generate_answer(
                user_question=user_input,
                chat_history=groq_history
            )

        st.markdown(result["answer"])

        with st.expander("📚 Sources used", expanded=False):
            for source_id in result["sources"]:
                st.caption(f"• {source_id}")

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": result["answer"]
    })