# app.py
# PURPOSE: Streamlit chat interface for the Clinic RAG Chatbot.
# This is Stage 7 — the final stage of our RAG pipeline.
#
# Run with:  streamlit run app.py

import streamlit as st
import sys
import os

# Make sure Python can find our src/ modules
sys.path.append(os.path.dirname(__file__))

from src.generator import generate_answer

# ── PAGE CONFIGURATION ───────────────────────────────────────────────────────
# This must be the FIRST Streamlit command in the file
st.set_page_config(
    page_title="Clinic Assistant",
    page_icon="🏥",
    layout="centered"
)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("🏥 Clinic Assistant")
st.caption(
    "Ask me anything about Sunrise Medical Center, "
    "Green Leaf Dental, or Wellness Eye Clinic."
)
st.divider()

# ── SESSION STATE SETUP ──────────────────────────────────────────────────────
# IMPORTANT CONCEPT: Streamlit re-runs the ENTIRE script on every
# user interaction. Without session_state, all variables reset to zero
# on every message — your chat history would vanish!
#
# st.session_state works like a "memory" that survives re-runs.
# Think of it as a sticky notepad that Streamlit keeps between runs.

if "messages" not in st.session_state:
    # First time the app loads — initialize with a welcome message
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

# ── DISPLAY CHAT HISTORY ─────────────────────────────────────────────────────
# Loop through all messages stored in session_state and display them.
# st.chat_message("user")      → shows message on the RIGHT with user icon
# st.chat_message("assistant") → shows message on the LEFT with bot icon

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── HANDLE NEW USER INPUT ────────────────────────────────────────────────────
# st.chat_input() shows the text box at the bottom of the screen.
# It returns the typed text when user presses Enter, otherwise None.

user_input = st.chat_input("Ask about doctors, services, timings...")

if user_input:

    # 1. Display user message immediately in the chat
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Save user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # 3. Build chat history for Groq (exclude the welcome message)
    #    We skip the first message (the welcome) and the current one
    #    (already added above). We only pass the middle history.
    #    Also exclude source lines — those are for display only.
    groq_history = []
    for msg in st.session_state.messages[1:-1]:  # skip first & last
        if msg["role"] in ["user", "assistant"]:
            groq_history.append({
                "role"   : msg["role"],
                "content": msg["content"]
            })

    # 4. Show a spinner while waiting for the response
    with st.chat_message("assistant"):
        with st.spinner("Searching clinic records..."):

            # ── FULL RAG PIPELINE RUNS HERE ──
            result = generate_answer(
                user_question=user_input,
                chat_history=groq_history
            )

        # 5. Display the answer
        st.markdown(result["answer"])

        # 6. Show sources in a subtle expander (collapsed by default)
        #    This is a transparency feature — users can see where
        #    the information came from
        with st.expander("📚 Sources used", expanded=False):
            for source_id in result["sources"]:
                st.caption(f"• {source_id}")

    # 7. Save assistant response to history
    st.session_state.messages.append({
        "role"   : "assistant",
        "content": result["answer"]
    })