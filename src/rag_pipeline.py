"""Retrieval-grounded Medical VQA pipeline."""

from __future__ import annotations

from typing import Any

from src.answer_normalization import answers_match, normalize_answer
from src.baseline import generate_from_prompt
from src.data_loader import VQAExample
from src.model_loader import VLMSession
from src.prompts import build_rag_prompt
from src.retrieval import Retriever


def retrieve_for_question(retriever: Retriever, question: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Retrieve evidence independently of generation so it can be tested alone."""
    return [doc.to_dict() for doc in retriever.retrieve(question, top_k=top_k)]


def run_rag_example(
    session: VLMSession,
    retriever: Retriever,
    example: VQAExample,
    top_k: int = 3,
    max_new_tokens: int = 50,
) -> dict[str, Any]:
    """Answer one VQA example with retrieved medical passages."""
    record: dict[str, Any] = {
        "example_id": example.example_id,
        "split": example.split,
        "index": example.index,
        "question": example.question,
        "ground_truth": example.answer,
        "system": "rag",
        "model_id": session.model_id,
        "device": session.device,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "failed": False,
        "error": "",
    }
    retrieved: list[dict[str, Any]] = []
    try:
        retrieved = retrieve_for_question(retriever, example.question, top_k=top_k)
        prompt = build_rag_prompt(example.question, retrieved)
        output = generate_from_prompt(session, example.image, prompt, max_new_tokens=max_new_tokens)
        raw = output["raw_answer"]
        record.update(
            {
                "prompt": prompt,
                "raw_decoded": output["raw_decoded"],
                "raw_answer": raw,
                "normalized_prediction": normalize_answer(raw),
                "normalized_ground_truth": normalize_answer(example.answer),
                "correct": answers_match(raw, example.answer),
                "inference_seconds": output["inference_seconds"],
                "retrieved_documents": retrieved,
                "retrieved_ids": [doc["id"] for doc in retrieved],
                "retrieval_l2_distances": [doc["score_l2_distance"] for doc in retrieved],
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
                "retrieved_documents": retrieved,
                "retrieved_ids": [doc["id"] for doc in retrieved],
            }
        )
    return record
