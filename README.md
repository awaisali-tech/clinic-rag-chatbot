# 🏥 Clinic RAG Chatbot

An AI-powered clinic assistant built using **Retrieval-Augmented Generation (RAG)** architecture. Patients can ask questions about doctors, services, timings, and FAQs across multiple clinics — and get accurate, sourced answers instantly.

🔗 **Live Demo:** [Click here to try it](https://clinic-rag-chatbot-bqdyrpp3bqnctsunp5bbrx.streamlit.app)

---

## 📸 Preview

> Ask questions like:
> - *"What are Dr. Ayesha's working hours?"*
> - *"Which clinic should I visit for a root canal?"*
> - *"Does Wellness Eye Clinic sell glasses?"*
> - *"Can I book an appointment on WhatsApp?"*

---

## 🧠 How It Works (RAG Pipeline)


clinic_data.json
↓
Stage 1: Data Loading      → Load clinic JSON data
↓
Stage 2: Chunking          → Split into 21 meaningful text chunks
↓
Stage 3: Embeddings        → Convert chunks to vectors (numbers)
↓
Stage 4: ChromaDB          → Store vectors in vector database
↓
[User asks a question]
↓
Stage 5: Retrieval         → Find top 3 most relevant chunks
↓
Stage 6: Groq LLM          → Generate accurate answer from context
↓
Stage 7: Streamlit UI      → Display answer in chat interface

---

## 🏗️ Project Structure

clinic-rag-chatbot/
├── data/
│   └── clinic_data.json        # Clinic data (doctors, services, FAQs)
├── src/
│   ├── data_loader.py          # Stage 1: Load JSON data
│   ├── chunker.py              # Stage 2: Create text chunks
│   ├── embedder.py             # Stage 3+4: Embed + store in ChromaDB
│   ├── retriever.py            # Stage 5: Search ChromaDB
│   └── generator.py            # Stage 6: Generate answer via Groq
├── app.py                      # Stage 7: Streamlit chat interface
├── ingest.py                   # One-time data ingestion script
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Chat Interface | Streamlit |
| Vector Database | ChromaDB |
| Embeddings | ChromaDB DefaultEmbeddingFunction (all-MiniLM-L6-v2) |
| LLM | Groq (LLaMA 3.3 70B) |
| Hosting | Streamlit Cloud |
| Version Control | GitHub |

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/awaisali-tech/clinic-rag-chatbot.git
cd clinic-rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Set up API key
Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here

Get your free Groq API key at: https://console.groq.com

### 5. Ingest clinic data (run once)
```bash
python ingest.py
```

### 6. Launch the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 💬 Example Questions

| Question | Expected Answer |
|----------|----------------|
| What are Dr. Ayesha's hours? | Mon-Fri 9am-2pm at Sunrise Medical |
| Which clinic for root canal? | Green Leaf Dental & Oral Care |
| Can I get LASIK surgery? | Yes, at Wellness Eye Clinic |
| Do you accept walk-ins? | Yes, but appointments recommended |
| What payment methods do you accept? | Cash, cards, bank transfer |

---

## 🏥 Clinics Covered

- 🏥 **Sunrise Medical Center** — General, Pediatrics, Dermatology, Cardiology, Lab Tests
- 🦷 **Green Leaf Dental & Oral Care** — Orthodontics, Root Canal, Emergency Dental
- 👁️ **Wellness Eye Clinic** — Eye Exams, LASIK, Cataract Surgery, Optical Shop

---

## 🔒 Security Features

- API keys stored in `.env` file (never committed to GitHub)
- `.gitignore` protects sensitive files
- Streamlit Secrets Manager used for cloud deployment

---

## 👨‍💻 Author

**Awais Ali**
- GitHub: [@awaisali-tech](https://github.com/awaisali-tech)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).