from datasets import load_dataset
from transformers import SmolVLMProcessor, SmolVLMForConditionalGeneration
import torch
import pandas as pd


# ----------------------------------------
# ANSWER NORMALIZATION
# ----------------------------------------

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


# ----------------------------------------
# LOAD DATASET
# ----------------------------------------

print("Loading VQA-RAD dataset...")

ds = load_dataset("flaviagiammarino/vqa-rad")

print("Dataset loaded!")

test_dataset = ds["test"]

print("Number of test examples:", len(test_dataset))


# ----------------------------------------
# LOAD SMOLVLM
# ----------------------------------------

print("\nLoading SmolVLM2...")

model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = SmolVLMProcessor.from_pretrained(
    model_name
)

model = SmolVLMForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = model.to(device)

print("Using device:", device)
print("Model loaded successfully!")


# ----------------------------------------
# RUN FULL TEST
# ----------------------------------------

print("\nStarting full SmolVLM evaluation...")

results = []

correct = 0

total = 25


for i in range(total):

    example = test_dataset[i]

    image = example["image"]
    question = example["question"]
    ground_truth = example["answer"]


    # ----------------------------------------
    # CREATE PROMPT
    # ----------------------------------------

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


    # ----------------------------------------
    # PROCESS IMAGE + QUESTION
    # ----------------------------------------

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    )


    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


    # ----------------------------------------
    # GENERATE ANSWER
    # ----------------------------------------

    output = model.generate(
        **inputs,
        max_new_tokens=20
    )


    # Remove the original prompt tokens
    generated_tokens = output[
        :,
        inputs["input_ids"].shape[1]:
    ]


    prediction = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True
    )[0].strip()


    # ----------------------------------------
    # NORMALIZE ANSWERS
    # ----------------------------------------

    normalized_prediction = normalize_answer(
        prediction
    )

    normalized_ground_truth = normalize_answer(
        ground_truth
    )


    # ----------------------------------------
    # CHECK CORRECTNESS
    # ----------------------------------------

    is_correct = (
        normalized_prediction
        == normalized_ground_truth
    )


    if is_correct:
        correct += 1


    # ----------------------------------------
    # SAVE RESULT
    # ----------------------------------------


    results.append(
        {
            "index": i,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "normalized_ground_truth": normalized_ground_truth,
            "normalized_prediction": normalized_prediction,
            "correct": is_correct
        }
    )

    # Save progress after every example
    pd.DataFrame(results).to_csv(
        "smolvlm_baseline_results.csv",
        index=False
    )

    # ----------------------------------------
    # SHOW PROGRESS
    # ----------------------------------------

    if (i + 1) % 10 == 0:

        current_accuracy = (
            correct / (i + 1)
        ) * 100

        print(
            f"Processed {i + 1}/{total} "
            f"| Correct: {correct} "
            f"| Accuracy: {current_accuracy:.2f}%"
        )


# ----------------------------------------
# FINAL RESULTS
# ----------------------------------------

accuracy = (
    correct / total
) * 100


print("\n========================================")
print("SMOLVLM FULL BASELINE")
print("========================================")

print("Total:", total)
print("Correct:", correct)
print("Incorrect:", total - correct)
print("Accuracy:", accuracy, "%")

print("========================================")


# ----------------------------------------
# SAVE CSV
# ----------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "smolvlm_baseline_results.csv",
    index=False
)

print(
    "\nResults saved to: "
    "smolvlm_baseline_results.csv"
)