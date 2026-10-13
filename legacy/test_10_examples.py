from datasets import load_dataset
from transformers import BlipProcessor, BlipForQuestionAnswering
import pandas as pd

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

print("\nStarting full baseline evaluation...")
print("Number of test examples:", len(ds["test"]))

results = []

correct = 0
total = len(ds["test"])

for i in range(total):

    example = ds["test"][i]

    image = example["image"]
    question = example["question"]
    correct_answer = example["answer"]

    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )

    output = model.generate(
        **inputs,
        max_new_tokens=10
    )

    predicted_answer = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    is_correct = (
        predicted_answer.strip().lower()
        == correct_answer.strip().lower()
    )

    if is_correct:
        correct += 1

    results.append({
        "example": i + 1,
        "question": question,
        "ground_truth": correct_answer,
        "prediction": predicted_answer,
        "correct": is_correct
    })

    print(
        f"Example {i + 1}/{total} | "
        f"Prediction: {predicted_answer} | "
        f"Correct: {is_correct}"
    )

accuracy = (correct / total) * 100

print("\n======================================")
print("FULL BASELINE EVALUATION")
print("======================================")
print("Total examples:", total)
print("Correct:", correct)
print("Incorrect:", total - correct)
print("Accuracy:", accuracy, "%")
print("======================================")

df = pd.DataFrame(results)

df.to_csv(
    "baseline_results.csv",
    index=False
)

print("\nResults saved to:")
print("baseline_results.csv")