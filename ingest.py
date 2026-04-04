# ingest.py
# PURPOSE: One-time data ingestion script.
# Run this ONCE to load JSON → chunk → embed → store in ChromaDB.
# After this, your data is saved permanently. Don't run again unless
# you change clinic_data.json (delete chroma_db/ folder first if so).

import os
import sys

# Make sure Python can find our src/ modules
sys.path.append(os.path.dirname(__file__))

from src.data_loader import load_clinic_data
from src.chunker     import create_chunks
from src.embedder    import ingest_chunks

def main():
    print("=" * 50)
    print("   CLINIC RAG — DATA INGESTION PIPELINE")
    print("=" * 50)

    # ── STAGE 1: Load JSON ───────────────────────────────
    print("\n[Stage 1] Loading clinic data...")
    data_path = os.path.join(os.path.dirname(__file__), "data", "clinic_data.json")
    clinic_data = load_clinic_data(data_path)

    # ── STAGE 2: Chunk ───────────────────────────────────
    print("\n[Stage 2] Creating chunks...")
    chunks = create_chunks(clinic_data)
    print(f"✅ Created {len(chunks)} chunks")

    # Preview 1 chunk so you can see what's being stored
    print(f"\n   Preview of first chunk:")
    print(f"   ID   : {chunks[0]['id']}")
    print(f"   Text : {chunks[0]['text'][:80]}...")  # first 80 characters

    # ── STAGE 3+4: Embed + Store ─────────────────────────
    print("\n[Stage 3+4] Embedding and storing in ChromaDB...")
    ingest_chunks(chunks)

    print("\n" + "=" * 50)
    print("   INGESTION COMPLETE — Ready for queries!")
    print("=" * 50)

if __name__ == "__main__":
    main()