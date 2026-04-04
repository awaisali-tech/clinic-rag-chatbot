# test_embeddings.py
# Testing ChromaDB's built-in embedding function
# This uses onnxruntime under the hood — no PyTorch needed!

from chromadb.utils import embedding_functions

# This uses the same all-MiniLM-L6-v2 model but via onnxruntime
# First run downloads ~25MB model — much smaller than PyTorch!
print("Loading embedding model... (first run downloads ~25MB)")

embedder = embedding_functions.DefaultEmbeddingFunction()

# Test sentences
test_sentences = [
    "Dr. Ayesha Khan is available Monday to Friday.",
    "The clinic is closed on Sundays.",
    "We accept cash and credit cards."
]

# Generate embeddings
embeddings = embedder(test_sentences)

print(f"\n✅ Embedding works!")
print(f"Number of sentences embedded: {len(embeddings)}")
print(f"Embedding size per sentence: {len(embeddings[0])} numbers")
print(f"First 5 numbers of sentence 1: {embeddings[0][:5]}")