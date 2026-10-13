from datasets import load_dataset
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

print("Loading VQA-RAD dataset...")

ds = load_dataset("flaviagiammarino/vqa-rad")

print("Dataset loaded!")

print("\nLoading BLIP-VQA model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)

model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

print("Model loaded!")

# Take the first test example
example = ds["test"][0]

image = example["image"]
question = example["question"]
correct_answer = example["answer"]

print("\n--- VQA-RAD Example ---")
print("Question:", question)
print("Correct answer:", correct_answer)

# Prepare image and question
inputs = processor(
    images=image,
    text=question,
    return_tensors="pt"
)

# Generate model answer
output = model.generate(**inputs)

predicted_answer = processor.decode(
    output[0],
    skip_special_tokens=True
)

print("Predicted answer:", predicted_answer)