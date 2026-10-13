from datasets import load_dataset
from transformers import SmolVLMProcessor, SmolVLMForConditionalGeneration
import torch
def normalize_answer(answer):
    answer = answer.lower().strip()

    if answer.startswith("yes"):
        return "yes"

    if answer.startswith("no"):
        return "no"

    if "right side" in answer:
        return "right"

    if "left side" in answer:
        return "left"

    if "not seen" in answer:
        return "not seen here"

    return answer

print("Loading VQA-RAD dataset...")

ds = load_dataset("flaviagiammarino/vqa-rad")

print("Dataset loaded!")

print("\nLoading SmolVLM2...")

model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = SmolVLMProcessor.from_pretrained(model_name)

model = SmolVLMForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)

print("Using device:", device)
print("Model loaded successfully!")

print("\nStarting 5-example test...")

correct = 0

for i in range(5):

    example = ds["test"][i]

    image = example["image"]
    question = example["question"]
    ground_truth = example["answer"]

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

    output = model.generate(
        **inputs,
        max_new_tokens=10
    )

    generated_tokens = output[:, inputs["input_ids"].shape[1]:]

    prediction = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0].strip()

    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    is_correct = (
    normalized_prediction == normalized_ground_truth
    )


    if is_correct:
        correct += 1

    print("\n----------------------------------------")
    print("Example:", i + 1)
    print("Question:", question)
    print("Ground truth:", ground_truth)
    print("SmolVLM prediction:", prediction)
    print("Normalized prediction:", normalized_prediction)
    print("Correct:", is_correct)

print("\n========================================")
print("SMOLVLM 5-EXAMPLE BASELINE")
print("========================================")
print("Total:", 5)
print("Correct:", correct)
print("Incorrect:", 5 - correct)
print("Accuracy:", (correct / 5) * 100, "%")
print("========================================")