import json
from sentence_transformers import SentenceTransformer

print("Loading medical knowledge base...")

with open(
    "data/knowledge_base/medical_knowledge.json",
    "r",
    encoding="utf-8"
) as file:
    knowledge_base = json.load(file)

print("Knowledge base loaded!")

print("\nLoading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding model loaded!")

texts = []

for document in knowledge_base:
    texts.append(document["text"])

print("\nCreating embeddings...")

embeddings = model.encode(texts)

print("Embeddings created successfully!")

print("\nNumber of documents:", len(texts))
print("Embedding shape:", embeddings.shape)

print("\nFirst document:")
print(texts[0])

print("\nFirst embedding:")
print(embeddings[0])