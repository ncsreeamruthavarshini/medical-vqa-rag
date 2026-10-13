from transformers import SmolVLMProcessor, SmolVLMForConditionalGeneration
from PIL import Image
import torch

print("Loading SmolVLM2...")

model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = SmolVLMProcessor.from_pretrained(model_name)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = SmolVLMForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

model = model.to(device)

print("Using device:", device)
print("Model loaded successfully!")

image = Image.open("test_image.jpg").convert("RGB")

question = "What is shown in this image?"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image"
            },
            {
                "type": "text",
                "text": question
            }
        ]
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True
)

inputs = processor(
    text=prompt,
    images=[image],
    return_tensors="pt"
)

inputs = {
    key: value.to(device) if hasattr(value, "to") else value
    for key, value in inputs.items()
}

print("Generating answer...")

output = model.generate(
    **inputs,
    max_new_tokens=20
)

answer = processor.batch_decode(
    output,
    skip_special_tokens=True
)[0]

print("\nQuestion:", question)
print("Answer:", answer)