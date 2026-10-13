import json
import faiss
import numpy as np

from datasets import load_dataset
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

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# -----------------------------------------
# Create embeddings for medical documents
# -----------------------------------------

texts = []

for document in knowledge_base:
    texts.append(document["text"])

print("\nCreating medical document embeddings...")

document_embeddings = model.encode(texts)

document_embeddings = np.array(
    document_embeddings
).astype("float32")

print("Document embeddings created!")


# -----------------------------------------
# Create FAISS index
# -----------------------------------------

print("\nCreating FAISS index...")

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(document_embeddings)

print("FAISS index created!")
print("Documents in index:", index.ntotal)


# -----------------------------------------
# Load VQA-RAD
# -----------------------------------------

print("\nLoading VQA-RAD dataset...")

ds = load_dataset(
    "flaviagiammarino/vqa-rad"
)

print("VQA-RAD loaded!")

test_data = ds["test"]


# -----------------------------------------
# Retrieve evidence for 5 questions
# -----------------------------------------

print("\n========================================")
print("REAL VQA-RAD RETRIEVAL TEST")
print("========================================")


number_of_questions = 5

for i in range(number_of_questions):

    example = test_data[i]

    question = example["question"]
    correct_answer = example["answer"]

    print("\n----------------------------------------")
    print("Example:", i + 1)

    print("\nQuestion:")
    print(question)

    print("\nGround-truth answer:")
    print(correct_answer)


    # Convert question into embedding

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")


    # Retrieve top 3 documents

    k = 3

    distances, indices = index.search(
        question_embedding,
        k
    )


    print("\nRetrieved medical evidence:")

    for rank in range(k):

        document_index = indices[0][rank]

        distance = distances[0][rank]

        document = knowledge_base[document_index]

        print("\nRank:", rank + 1)
        print("Topic:", document["topic"])
        print("Distance:", distance)
        print("Evidence:", document["text"])


print("\n========================================")
print("RETRIEVAL TEST COMPLETE")
print("========================================")
