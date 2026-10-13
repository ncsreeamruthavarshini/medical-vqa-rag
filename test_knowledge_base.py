import json

print("Loading medical knowledge base...")

with open(
    "data/knowledge_base/medical_knowledge.json",
    "r",
    encoding="utf-8"
) as file:

    knowledge_base = json.load(file)

print("Knowledge base loaded successfully!")

print("\nNumber of documents:", len(knowledge_base))

print("\nFirst document:")
print("ID:", knowledge_base[0]["id"])
print("Topic:", knowledge_base[0]["topic"])
print("Text:", knowledge_base[0]["text"])