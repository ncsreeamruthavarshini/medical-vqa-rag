"""Automatic evaluation metrics.

Exact-match accuracy after normalization is a research proxy, not clinical
accuracy. Hallucination rates are not computed here; those require human
review using the protocol in ``docs/methodology.md``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from src.answer_normalization import answers_match, is_yes_no, normalize_answer


def exact_match_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> dict[str, Any]:
    """Compute normalized exact-match accuracy on aligned sequences."""
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")
    n = len(predictions)
    if n == 0:
        return {
            "n": 0,
            "correct": 0,
            "incorrect": 0,
            "exact_match_accuracy": None,
            "denominator": 0,
        }
    correct = sum(1 for pred, gold in zip(predictions, ground_truths) if answers_match(pred, gold))
    return {
        "n": n,
        "correct": correct,
        "incorrect": n - correct,
        "exact_match_accuracy": correct / n,
        "denominator": n,
        "denominator_description": "all provided pairs",
    }


def yes_no_accuracy(predictions: Sequence[str], ground_truths: Sequence[str]) -> dict[str, Any]:
    """Exact-match accuracy restricted to items whose gold answer is yes or no."""
    pairs = [
        (pred, gold)
        for pred, gold in zip(predictions, ground_truths)
        if is_yes_no(gold)
    ]
    if not pairs:
        return {
            "n_yes_no": 0,
            "correct": 0,
            "yes_no_accuracy": None,
            "denominator_description": "items whose normalized gold answer is exactly yes or no",
        }
    correct = sum(1 for pred, gold in pairs if answers_match(pred, gold))
    n = len(pairs)
    return {
        "n_yes_no": n,
        "correct": correct,
        "incorrect": n - correct,
        "yes_no_accuracy": correct / n,
        "denominator_description": "items whose normalized gold answer is exactly yes or no",
    }


def binomial_standard_error(accuracy: float | None, n: int) -> float | None:
    """Standard error of a proportion. Not a claim of statistical significance."""
    if accuracy is None or n <= 0:
        return None
    return float((accuracy * (1.0 - accuracy) / n) ** 0.5)


def recall_at_k(
    retrieved_ids: Sequence[Sequence[str]],
    relevant_ids: Sequence[Iterable[str]],
    k: int,
) -> dict[str, Any]:
    """Recall@k over queries that have at least one relevant document id.

    Labels must be provided by a human. Do not invent them.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if len(retrieved_ids) != len(relevant_ids):
        raise ValueError("retrieved_ids and relevant_ids must have the same length")

    scores: list[float] = []
    skipped = 0
    for retrieved, relevant in zip(retrieved_ids, relevant_ids):
        relevant_set = {str(item) for item in relevant}
        if not relevant_set:
            skipped += 1
            continue
        top = [str(item) for item in retrieved[:k]]
        hits = sum(1 for doc_id in relevant_set if doc_id in top)
        scores.append(hits / len(relevant_set))

    n = len(scores)
    return {
        "k": k,
        "n_labeled_queries": n,
        "n_skipped_unlabeled": skipped,
        "recall_at_k": (sum(scores) / n) if n else None,
        "denominator_description": "queries with at least one human-assigned relevant document id",
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize prediction records that include match and error fields."""
    evaluated = [row for row in records if not row.get("failed")]
    failed = [row for row in records if row.get("failed")]
    predictions = [str(row.get("normalized_prediction", "")) for row in evaluated]
    golds = [str(row.get("normalized_ground_truth", "")) for row in evaluated]
    em = exact_match_accuracy(predictions, golds)
    yn = yes_no_accuracy(predictions, golds)
    error_kinds = Counter(str(row.get("error") or "none") for row in failed)
    return {
        "n_records": len(records),
        "n_evaluated": em["n"],
        "n_failed": len(failed),
        "n_correct": em["correct"],
        "n_incorrect": em["incorrect"],
        "exact_match_accuracy": em["exact_match_accuracy"],
        "accuracy_standard_error": binomial_standard_error(em["exact_match_accuracy"], em["n"]),
        "yes_no": yn,
        "failed_error_counts": dict(error_kinds),
        "denominator_description": "successful inferences only; failed examples are reported separately",
    }


def paired_outcomes(
    baseline_correct: Sequence[bool],
    rag_correct: Sequence[bool],
) -> dict[str, Any]:
    """Count paired correctness outcomes for the same examples."""
    if len(baseline_correct) != len(rag_correct):
        raise ValueError("paired sequences must have the same length")
    both = baseline_only = rag_only = neither = 0
    for base, rag in zip(baseline_correct, rag_correct):
        if base and rag:
            both += 1
        elif base and not rag:
            baseline_only += 1
        elif rag and not base:
            rag_only += 1
        else:
            neither += 1
    n = len(baseline_correct)
    return {
        "n_paired": n,
        "both_correct": both,
        "baseline_only_correct": baseline_only,
        "rag_only_correct": rag_only,
        "both_incorrect": neither,
    }
