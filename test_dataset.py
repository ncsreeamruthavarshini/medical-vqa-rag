from datasets import load_dataset

print("Loading VQA-RAD...")

ds = load_dataset("flaviagiammarino/vqa-rad")

print("\nDataset loaded successfully!")
print(ds)

print("\nTrain columns:")
print(ds["train"].column_names)

print("\nFirst training example:")
print(ds["train"][0])