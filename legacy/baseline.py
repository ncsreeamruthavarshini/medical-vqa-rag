from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load BLIP-VQA
print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-vqa-base"
)

model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base"
)

print("Model loaded successfully!")

# Load image
image = Image.open("test_image.jpg").convert("RGB")

# Question
question = "What is shown in the image?"

# Prepare input
inputs = processor(
    images=image,
    text=question,
    return_tensors="pt"
)

# Generate answer
output = model.generate(**inputs)

# Convert output to text
answer = processor.decode(
    output[0],
    skip_special_tokens=True
)

print("\nQuestion:", question)
print("Answer:", answer)