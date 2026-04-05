# app.py
# Stage 7: Streamlit Chat Interface
# The entire app — clean, correct, true RAG architecture

import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))


@st.cache_resource
def initialize_rag():
    """
    Initializes the FULL RAG pipeline ONCE.
    Returns the ChromaDB collection — shared across ALL users.

    @st.cache_resource guarantees:
    - Runs only ONCE per app session
    - Same object returned to every user
    - Survives Streamlit reruns
    """
    from src.data_loader import load_clinic_data
    from src.chunker     import create_chunks
    from src.embedder    import get_chroma_client, get_chroma_collection, ingest_chunks

    print("🚀 Initializing RAG pipeline...")

    # Stage 1: Load data
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "clinic_data.json"
    )
    clinic_data = load_clinic_data(data_path)

    # Stage 2: Chunk
    chunks = create_chunks(clinic_data)

    # Stage 3+4: Embed + Store
    client     = get_chroma_client()
    collection = get_chroma_collection(client)

    if collection.count() == 0:
        ingest_chunks(chunks, client)

    print(f"✅ RAG ready — {collection.count()} chunks in ChromaDB")
    return collection


# Initialize once — shared across ALL users ✅
collection = initialize_rag()

from src.generator import generate_answer

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
st.sidebar.write(f"✅ ChromaDB chunks: {collection.count()}")
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
                collection=collection,
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