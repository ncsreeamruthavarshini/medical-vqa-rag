import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading medical knowledge base...")

with open(
    "data/knowledge_base/medical_knowledge.json",
    "r",
    encoding="utf-8"
) as file:
    knowledge_base = json.load(file)

print("Knowledge base loaded!")

print("\nLoading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")

# Get the text from every medical document
texts = []

for document in knowledge_base:
    texts.append(document["text"])

print("\nCreating document embeddings...")

document_embeddings = model.encode(texts)

print("Document embeddings created!")

# Convert embeddings into the format FAISS expects
document_embeddings = np.array(
    document_embeddings
).astype("float32")

print("\nCreating FAISS index...")

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

print("FAISS index created!")
print("Number of documents in index:", index.ntotal)

# -----------------------------------------
# Test question
# -----------------------------------------

questions = [
    "Is there airspace consolidation in the lung?",
    "Where are the lateral ventricles located?",
    "Where are the kidneys located?",
    "What structures can be seen on a chest radiograph?",
    "What is a CT scan?"
]

print("\n========================================")
print("RETRIEVAL TEST")
print("========================================")

# Test every question
for question in questions:

    print("\n\nQuestion:")
    print(question)

    print("\nConverting question into embedding...")

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    # Search for the 3 closest documents
    k = 3

    distances, indices = index.search(
        question_embedding,
        k
    )

    print("\nTop", k, "retrieved documents:")

    for rank in range(k):

        document_index = indices[0][rank]
        distance = distances[0][rank]

        document = knowledge_base[document_index]

        print("\n----------------------------------------")
        print("Rank:", rank + 1)
        print("Document ID:", document["id"])
        print("Topic:", document["topic"])
        print("Distance:", distance)
        print("Text:", document["text"])

print("\n========================================")
print("RETRIEVAL COMPLETE")
print("========================================")