"""Baseline Medical VQA: image + question -> vision-language model -> answer."""

from __future__ import annotations

import time
from typing import Any

from PIL import Image

from src.answer_normalization import answers_match, normalize_answer
from src.data_loader import VQAExample
from src.model_loader import VLMSession, extract_generated_text, move_batch_to_device
from src.prompts import build_baseline_prompt, build_conversation


def generate_from_prompt(
    session: VLMSession,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 50,
) -> dict[str, Any]:
    """Run one VLM generation and return the raw and extracted answer."""
    if image is None:
        raise ValueError("An image is required.")
    if not isinstance(image, Image.Image):
        raise ValueError("image must be a PIL Image.")
    image = image.convert("RGB")

    messages = build_conversation(prompt)
    text = session.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = session.processor(text=text, images=[image], return_tensors="pt")
    inputs = move_batch_to_device(inputs, session.device)

    started = time.perf_counter()
    import torch

    with torch.no_grad():
        generated_ids = session.model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - started

    raw_full = session.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    answer = extract_generated_text(
        session.processor,
        generated_ids,
        input_ids=inputs.get("input_ids"),
    )
    return {
        "prompt": prompt,
        "raw_decoded": str(raw_full),
        "raw_answer": answer,
        "inference_seconds": elapsed,
    }


def run_baseline_example(
    session: VLMSession,
    example: VQAExample,
    max_new_tokens: int = 50,
) -> dict[str, Any]:
    """Answer one VQA example without retrieved evidence."""
    record: dict[str, Any] = {
        "example_id": example.example_id,
        "split": example.split,
        "index": example.index,
        "question": example.question,
        "ground_truth": example.answer,
        "system": "baseline",
        "model_id": session.model_id,
        "device": session.device,
        "max_new_tokens": max_new_tokens,
        "failed": False,
        "error": "",
    }
    try:
        prompt = build_baseline_prompt(example.question)
        output = generate_from_prompt(session, example.image, prompt, max_new_tokens=max_new_tokens)
        raw = output["raw_answer"]
        normalized_pred = normalize_answer(raw)
        normalized_gold = normalize_answer(example.answer)
        record.update(
            {
                "prompt": prompt,
                "raw_decoded": output["raw_decoded"],
                "raw_answer": raw,
                "normalized_prediction": normalized_pred,
                "normalized_ground_truth": normalized_gold,
                "correct": answers_match(raw, example.answer),
                "inference_seconds": output["inference_seconds"],
            }
        )
    except Exception as exc:
        record.update(
            {
                "failed": True,
                "error": f"{type(exc).__name__}: {exc}",
                "raw_answer": "",
                "normalized_prediction": "",
                "normalized_ground_truth": normalize_answer(example.answer),
                "correct": False,
                "inference_seconds": None,
            }
        )
    return record
