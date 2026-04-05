# src/chunker.py
# PURPOSE: Convert raw clinic JSON data into clean, searchable text chunks.
# Each chunk = one self-contained piece of information about the clinic.
# This is Stage 2 of our RAG pipeline.

def create_chunks(clinic_data: dict) -> list[dict]:
    """
    Takes the full clinic data dictionary and returns a list of chunks.

    Each chunk is a dictionary with two keys:
        - "id"   : a unique name for this chunk (used by ChromaDB)
        - "text" : the actual text content that gets embedded and searched

    Args:
        clinic_data: The dictionary returned by load_clinic_data()

    Returns:
        A list of chunk dictionaries
    """
    chunks = []  # We'll collect all chunks here

    # Loop through each clinic in the JSON
    for clinic in clinic_data["clinics"]:

        clinic_id   = clinic["id"]      # e.g. "clinic_001"
        clinic_name = clinic["name"]    # e.g. "Sunrise Medical Center"

        # ── CHUNK TYPE 1: OVERVIEW ──────────────────────────────────────
        # One chunk with the essential clinic info
        # We join the services list into a comma-separated string

        services_text = ", ".join(clinic["services"])

        # Timings: convert the dict into readable lines
        # e.g. {"Mon-Fri": "9am-6pm"} → "Mon-Fri: 9am-6pm"
        timings_lines = "\n  ".join(
            [f"{day}: {hours}" for day, hours in clinic["timings"].items()]
        )

        overview_chunk = {
            "id": f"{clinic_id}_overview",
            "text": (
                f"Clinic Name: {clinic_name}\n"
                f"Address: {clinic['location']['address']}\n"
                f"Phone: {clinic['contact']['phone']}\n"
                f"Email: {clinic['contact']['email']}\n"
                f"Services Offered: {services_text}\n"
                f"Opening Hours (Monday=Mon, Tuesday=Tue, Wednesday=Wed, Thursday=Thu, Friday=Fri, Saturday=Sat, Sunday=Sun):\n  {timings_lines}"
            )
        }
        chunks.append(overview_chunk)

        # ── CHUNK TYPE 2: DOCTORS ───────────────────────────────────────
        # One chunk per doctor so retrieval is precise
        # e.g. "Which doctors work Monday?" → finds relevant doctor chunks

        for i, doctor in enumerate(clinic["doctors"]):
            doctor_chunk = {
                "id": f"{clinic_id}_doctor_{i+1}",
                "text": (
                    f"Clinic: {clinic_name}\n"
                    f"Doctor Name: {doctor['name']}\n"
                    f"Specialization: {doctor['specialization']}\n"
                    f"Experience: {doctor['experience_years']} years\n"
                    f"Availability: {doctor['availability']}"
                )
            }
            chunks.append(doctor_chunk)

        # ── CHUNK TYPE 3: FAQs ──────────────────────────────────────────
        # One chunk per FAQ — question AND answer together
        # This way, if someone asks something similar, we retrieve the answer

        for i, faq in enumerate(clinic["faqs"]):
            faq_chunk = {
                "id": f"{clinic_id}_faq_{i+1}",
                "text": (
                    f"Clinic: {clinic_name}\n"
                    f"Question: {faq['question']}\n"
                    f"Answer: {faq['answer']}"
                )
            }
            chunks.append(faq_chunk)

        # ── CHUNK TYPE 4: ABOUT ─────────────────────────────────────────
        # The clinic's background story / description paragraph

        about_chunk = {
            "id": f"{clinic_id}_about",
            "text": (
                f"About {clinic_name}:\n"
                f"{clinic['about']}"
            )
        }
        chunks.append(about_chunk)

    return chunks


# ── TEST BLOCK ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    # Add the project root to Python's path so we can import data_loader
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from src.data_loader import load_clinic_data

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "clinic_data.json")
    clinic_data = load_clinic_data(data_path)

    # Create chunks
    chunks = create_chunks(clinic_data)

    # Print summary
    print(f"\n✅ Total chunks created: {len(chunks)}")
    print(f"\n{'='*50}")
    print("SAMPLE CHUNKS (first 3):")
    print('='*50)

    for chunk in chunks[:3]:
        print(f"\n📄 Chunk ID : {chunk['id']}")
        print(f"📝 Content  :\n{chunk['text']}")
        print("-" * 40)