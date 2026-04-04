import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(__file__))


@st.cache_resource
def initialize_database():
    from src.data_loader import load_clinic_data
    from src.chunker import create_chunks
    from src.embedder import ingest_chunks, get_chunk_count

    if get_chunk_count() == 0:
        data_path = os.path.join(
            os.path.dirname(__file__), "data", "clinic_data.json"
        )
        clinic_data = load_clinic_data(data_path)
        chunks      = create_chunks(clinic_data)
        ingest_chunks(chunks)
    return True


initialize_database()

from src.generator import generate_answer

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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