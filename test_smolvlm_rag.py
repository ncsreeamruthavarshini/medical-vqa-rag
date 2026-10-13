
import json
import torch
import faiss

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForImageTextToText


# =========================================
# 1. LOAD MEDICAL KNOWLEDGE BASE
# =========================================

print("Loading medical knowledge base...")

with open(
    "data/knowledge_base/medical_knowledge.json",
    "r",
    encoding="utf-8"
) as file:
    knowledge_base = json.load(file)

if not knowledge_base:
    raise ValueError("The medical knowledge base is empty.")

print("Knowledge base loaded!")
print("Number of medical documents:", len(knowledge_base))


# =========================================
# 2. LOAD TEXT EMBEDDING MODEL
# =========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded!")


# =========================================
# 3. CREATE MEDICAL DOCUMENT EMBEDDINGS
# =========================================

print("\nCreating medical document embeddings...")

texts = [
    document["text"]
    for document in knowledge_base
]

document_embeddings = embedding_model.encode(
    texts,
    convert_to_numpy=True
).astype("float32")

print("Medical document embeddings created!")
print("Embedding shape:", document_embeddings.shape)


# =========================================
# 4. CREATE FAISS SEARCH INDEX
# =========================================

print("\nCreating FAISS index...")

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

print("FAISS index created!")
print("Number of documents in index:", index.ntotal)


# =========================================
# 5. LOAD VQA-RAD DATASET
# =========================================

print("\nLoading VQA-RAD dataset...")

dataset = load_dataset("flaviagiammarino/vqa-rad")
test_dataset = dataset["test"]

print("VQA-RAD test dataset loaded!")
print("Number of test examples:", len(test_dataset))


# =========================================
# 6. LOAD SMOLVLM2
# =========================================

print("\nLoading SmolVLM2...")

model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = AutoProcessor.from_pretrained(model_name)

model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", device)

model = model.to(device)
model.eval()

print("SmolVLM2 loaded successfully!")


# =========================================
# 7. RETRIEVE RELEVANT MEDICAL EVIDENCE
# =========================================

def retrieve_evidence(question, k=3):
    """
    Convert the question into an embedding and retrieve
    the most relevant medical documents using FAISS.
    """

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    # Never request more documents than the index contains.
    number_to_retrieve = min(k, index.ntotal)

    distances, indices = index.search(
        question_embedding,
        number_to_retrieve
    )

    retrieved_documents = []

    for rank in range(number_to_retrieve):
        document_index = int(indices[0][rank])

        if document_index < 0:
            continue

        document = knowledge_base[document_index]

        retrieved_documents.append({
            "id": document["id"],
            "topic": document["topic"],
            "text": document["text"],
            "distance": float(distances[0][rank])
        })

    return retrieved_documents


# =========================================
# 8. CREATE THE RAG PROMPT
# =========================================

def create_rag_prompt(question, retrieved_documents):
    """
    Combine the medical evidence and question into a prompt.
    """

    evidence_text = ""

    for document in retrieved_documents:
        evidence_text += (
            f"Medical evidence ({document['topic']}): "
            f"{document['text']}\n"
        )

    prompt = f"""
You are answering a medical visual question.

Examine the provided medical image and answer the question.
Use the retrieved medical evidence as supporting information,
but do not assume that general medical facts prove a finding
is present in this particular image.

If the image does not provide enough information, say so.
Do not invent findings or claim that the evidence confirms
something that cannot be seen in the image.

Retrieved medical evidence:
{evidence_text}

Question:
{question}

Give a concise answer based on the image and relevant evidence.
"""

    return prompt.strip()


# =========================================
# 9. GENERATE AN ANSWER USING RAG
# =========================================

def generate_rag_answer(image, question):
    """
    RAG pipeline:
    Question -> Retrieval -> Evidence
    Image + Question + Evidence -> SmolVLM2 -> Answer
    """

    # Step 1: Retrieve medical evidence.
    retrieved_documents = retrieve_evidence(
        question,
        k=3
    )

    # Step 2: Create a prompt containing the evidence.
    prompt = create_rag_prompt(
        question,
        retrieved_documents
    )

    # Step 3: Prepare the image and text conversation.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt"
    )

    # Step 4: Move model inputs to the model's device.
    inputs = {
        key: value.to(device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    # Step 5: Generate the answer.
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50
        )

    # Step 6: Decode the generated output.
    decoded_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    # Step 7: Extract the answer after the assistant marker.
    if "Assistant:" in decoded_output:
        answer = decoded_output.split("Assistant:")[-1].strip()
    else:
        answer = decoded_output.strip()

    return answer, retrieved_documents


# =========================================
# 10. TEST THE COMPLETE RAG PIPELINE
# =========================================

print("\n========================================")
print("RAG SINGLE EXAMPLE TEST")
print("========================================")

example = test_dataset[0]

image = example["image"]
question = example["question"]
correct_answer = example["answer"]

print("\nQuestion:")
print(question)

print("\nGround-truth answer:")
print(correct_answer)

print("\nGenerating RAG answer...")

answer, retrieved_documents = generate_rag_answer(
    image,
    question
)

print("\nModel answer:")
print(answer)

print("\nRetrieved evidence:")

for rank, document in enumerate(
    retrieved_documents,
    start=1
):
    print("\nRank:", rank)
    print("Document ID:", document["id"])
    print("Topic:", document["topic"])
    print("Distance:", document["distance"])
    print("Text:", document["text"])

print("\n========================================")
print("RAG TEST COMPLETE")
print("========================================")