"""Prompt construction for baseline and retrieval-grounded VQA."""

from __future__ import annotations

from typing import Mapping, Sequence

BASELINE_INSTRUCTION = (
    "You are answering a medical visual question for a research prototype. "
    "Inspect the image. Answer concisely, using a short phrase similar to "
    "dataset answers such as yes, no, left, right, or a brief finding. "
    "Do not invent findings that you cannot see. If the image does not "
    "contain enough information, say so."
)

RAG_INSTRUCTION = (
    "You are answering a medical visual question for a research prototype. "
    "Inspect the image first. Retrieved passages are general medical "
    "knowledge. They do not prove that a finding is present in this image. "
    "Use a passage only if it helps you interpret what you can actually see. "
    "Do not invent findings. Do not claim that retrieved text confirms an "
    "image finding by itself. If visual evidence is insufficient, say so. "
    "Give a concise answer consistent with short dataset-style responses."
)


def build_baseline_prompt(question: str) -> str:
    """Build the no-retrieval prompt."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must be a non-empty string.")
    return f"{BASELINE_INSTRUCTION}\n\nQuestion:\n{question}\n\nAnswer:"


def format_evidence(documents: Sequence[Mapping[str, object]]) -> str:
    """Turn retrieved documents into a numbered evidence block."""
    if not documents:
        return "No medical passages were retrieved."
    lines: list[str] = []
    for rank, doc in enumerate(documents, start=1):
        topic = str(doc.get("topic") or "unspecified")
        text = str(doc.get("text") or "").strip()
        source = str(doc.get("source_title") or "source not recorded")
        status = str(doc.get("status") or "")
        status_note = f" [{status}]" if status else ""
        lines.append(
            f"[{rank}] Topic: {topic}{status_note}\n"
            f"Source: {source}\n"
            f"Passage: {text}"
        )
    return "\n\n".join(lines)


def build_rag_prompt(question: str, documents: Sequence[Mapping[str, object]]) -> str:
    """Build the retrieval-grounded prompt."""
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must be a non-empty string.")
    evidence = format_evidence(documents)
    return (
        f"{RAG_INSTRUCTION}\n\n"
        "Retrieved medical evidence (general knowledge, not image proof):\n"
        f"{evidence}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def build_conversation(prompt: str) -> list[dict]:
    """SmolVLM2 chat format: one user turn with an image and text."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
