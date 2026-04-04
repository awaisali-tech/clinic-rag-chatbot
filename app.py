# app.py
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))


# ── SHARED DATABASE — initialized ONCE, shared across ALL users ──────────────
# @st.cache_resource is Streamlit's way of sharing one resource
# across all users and all sessions — perfect for a database connection!
@st.cache_resource
def get_shared_collection():
    """
    Creates ONE ChromaDB collection shared by everyone.
    Streamlit Cache Resource keeps this alive for the entire app lifetime.
    """
    import chromadb
    from chromadb.utils import embedding_functions
    from src.data_loader import load_clinic_data
    from src.chunker import create_chunks

    print("Initializing shared ChromaDB collection...")

    # Always use EphemeralClient on cloud
    is_cloud = not os.path.exists(
        os.path.join(os.path.dirname(__file__), "chroma_db")
    )

    if is_cloud:
        client = chromadb.EphemeralClient()
    else:
        from src.embedder import DB_PATH
        client = chromadb.PersistentClient(path=DB_PATH)

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="clinic_knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Ingest if empty
    if collection.count() == 0:
        print("Collection empty — ingesting data...")
        data_path = os.path.join(
            os.path.dirname(__file__), "data", "clinic_data.json"
        )
        clinic_data = load_clinic_data(data_path)
        chunks      = create_chunks(clinic_data)
        ids   = [c["id"]   for c in chunks]
        texts = [c["text"] for c in chunks]
        collection.add(ids=ids, documents=texts)
        print(f"✅ Ingested {len(chunks)} chunks!")

    print(f"Collection ready with {collection.count()} chunks ✅")
    return collection


# Get the shared collection — same instance for ALL users
collection = get_shared_collection()

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
                chat_history=groq_history,
                collection=collection    # ← pass shared collection
            )

        st.markdown(result["answer"])

        with st.expander("📚 Sources used", expanded=False):
            for source_id in result["sources"]:
                st.caption(f"• {source_id}")

    st.session_state.messages.append({
        "role"   : "assistant",
        "content": result["answer"]
    })