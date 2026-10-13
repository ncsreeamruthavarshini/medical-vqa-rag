"""Fair comparison of baseline and RAG prediction files."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.metrics import paired_outcomes, summarize_records


def load_prediction_csv(path: Path | str) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return [{key: value for key, value in row.items()} for row in csv.DictReader(handle)]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def find_id_problems(rows: Sequence[Mapping[str, Any]], id_field: str = "example_id") -> dict[str, Any]:
    ids = [str(row.get(id_field, "")) for row in rows]
    counts = Counter(ids)
    missing = [i for i, example_id in enumerate(ids) if example_id == ""]
    duplicated = sorted(example_id for example_id, count in counts.items() if example_id and count > 1)
    return {
        "n_rows": len(rows),
        "missing_id_row_indexes": missing,
        "duplicated_ids": duplicated,
    }


def compare_runs(
    baseline_rows: Sequence[Mapping[str, Any]],
    rag_rows: Sequence[Mapping[str, Any]],
    id_field: str = "example_id",
) -> dict[str, Any]:
    """Join baseline and RAG outputs by example id and score the same eligible set."""
    baseline_problems = find_id_problems(baseline_rows, id_field)
    rag_problems = find_id_problems(rag_rows, id_field)

    baseline_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rag_by_id: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        example_id = str(row.get(id_field, "")).strip()
        if example_id:
            baseline_by_id[example_id].append(row)
    for row in rag_rows:
        example_id = str(row.get(id_field, "")).strip()
        if example_id:
            rag_by_id[example_id].append(row)

    baseline_ids = set(baseline_by_id)
    rag_ids = set(rag_by_id)
    missing_in_rag = sorted(baseline_ids - rag_ids)
    missing_in_baseline = sorted(rag_ids - baseline_ids)
    shared_ids = sorted(baseline_ids & rag_ids)

    joined: list[dict[str, Any]] = []
    eligible_baseline: list[dict[str, Any]] = []
    eligible_rag: list[dict[str, Any]] = []
    skipped_failed: list[str] = []
    skipped_duplicate: list[str] = []

    for example_id in shared_ids:
        base_group = baseline_by_id[example_id]
        rag_group = rag_by_id[example_id]
        if len(base_group) != 1 or len(rag_group) != 1:
            skipped_duplicate.append(example_id)
            continue
        base = dict(base_group[0])
        rag = dict(rag_group[0])
        base_failed = _as_bool(base.get("failed", False))
        rag_failed = _as_bool(rag.get("failed", False))
        pair = {
            "example_id": example_id,
            "question": base.get("question") or rag.get("question"),
            "ground_truth": base.get("ground_truth") or rag.get("ground_truth"),
            "baseline_raw_answer": base.get("raw_answer"),
            "rag_raw_answer": rag.get("raw_answer"),
            "baseline_normalized": base.get("normalized_prediction"),
            "rag_normalized": rag.get("normalized_prediction"),
            "baseline_correct": _as_bool(base.get("correct", False)) if not base_failed else False,
            "rag_correct": _as_bool(rag.get("correct", False)) if not rag_failed else False,
            "baseline_failed": base_failed,
            "rag_failed": rag_failed,
            "baseline_error": base.get("error", ""),
            "rag_error": rag.get("error", ""),
            "retrieved_ids": rag.get("retrieved_ids", ""),
            "eligible": not (base_failed or rag_failed),
        }
        joined.append(pair)
        if pair["eligible"]:
            eligible_baseline.append(base)
            eligible_rag.append(rag)
        else:
            skipped_failed.append(example_id)

    baseline_summary = summarize_records(eligible_baseline)
    rag_summary = summarize_records(eligible_rag)
    paired = paired_outcomes(
        [_as_bool(row.get("correct", False)) for row in eligible_baseline],
        [_as_bool(row.get("correct", False)) for row in eligible_rag],
    )
    base_acc = baseline_summary.get("exact_match_accuracy")
    rag_acc = rag_summary.get("exact_match_accuracy")
    delta = None if base_acc is None or rag_acc is None else (rag_acc - base_acc)

    disagreement = [
        row
        for row in joined
        if row["eligible"] and row["baseline_correct"] != row["rag_correct"]
    ]

    return {
        "baseline_id_problems": baseline_problems,
        "rag_id_problems": rag_problems,
        "missing_in_rag": missing_in_rag,
        "missing_in_baseline": missing_in_baseline,
        "skipped_duplicate_ids": skipped_duplicate,
        "skipped_failed_ids": skipped_failed,
        "n_shared_ids": len(shared_ids),
        "n_eligible_pairs": len(eligible_baseline),
        "baseline": baseline_summary,
        "rag": rag_summary,
        "paired": paired,
        "accuracy_difference_rag_minus_baseline": delta,
        "joined_rows": joined,
        "disagreement_rows": disagreement,
        "denominator_description": (
            "Metrics use example ids present in both files, with exactly one row "
            "each, where neither inference failed. Failed inferences are listed "
            "separately and are not silently scored as incorrect in the paired accuracy."
        ),
    }


def format_comparison_report(result: Mapping[str, Any]) -> str:
    """Human-readable terminal report."""
    base = result["baseline"]
    rag = result["rag"]
    paired = result["paired"]
    delta = result["accuracy_difference_rag_minus_baseline"]
    lines = [
        "========================================",
        "BASELINE vs RAG COMPARISON",
        "========================================",
        f"Shared example ids: {result['n_shared_ids']}",
        f"Eligible paired examples: {result['n_eligible_pairs']}",
        f"Missing in RAG: {result['missing_in_rag']}",
        f"Missing in baseline: {result['missing_in_baseline']}",
        f"Skipped duplicates: {result['skipped_duplicate_ids']}",
        f"Skipped because either system failed: {result['skipped_failed_ids']}",
        "",
        f"Baseline exact-match accuracy: {_pct(base.get('exact_match_accuracy'))} "
        f"({base.get('n_correct')}/{base.get('n_evaluated')})",
        f"RAG exact-match accuracy: {_pct(rag.get('exact_match_accuracy'))} "
        f"({rag.get('n_correct')}/{rag.get('n_evaluated')})",
        f"Difference (RAG - baseline): {_pp(delta)}",
        "",
        f"Both correct: {paired.get('both_correct')}",
        f"Baseline only correct: {paired.get('baseline_only_correct')}",
        f"RAG only correct: {paired.get('rag_only_correct')}",
        f"Both incorrect: {paired.get('both_incorrect')}",
        "",
        "These figures are preliminary research metrics on a small sample.",
        "They are not clinical accuracy and do not prove hallucination reduction.",
        "========================================",
    ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.2f}%"


def _pp(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:+.2f} percentage points"


def save_comparison_outputs(result: Mapping[str, Any], output_dir: Path | str) -> None:
    from src.experiment_logger import write_csv, write_json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "comparison.csv", result["joined_rows"])
    write_csv(output_dir / "disagreements.csv", result["disagreement_rows"])
    summary = {key: value for key, value in result.items() if key not in {"joined_rows", "disagreement_rows"}}
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.txt").write_text(format_comparison_report(result), encoding="utf-8")
    _maybe_plot(result, output_dir / "accuracy_comparison.png")


def _maybe_plot(result: Mapping[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    base = result["baseline"].get("exact_match_accuracy") or 0.0
    rag = result["rag"].get("exact_match_accuracy") or 0.0
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.bar(["Baseline", "RAG"], [100 * base, 100 * rag], color=["#4C78A8", "#F58518"])
    axis.set_ylabel("Normalized exact-match accuracy (%)")
    axis.set_ylim(0, 100)
    n = result.get("n_eligible_pairs")
    axis.set_title(f"Preliminary accuracy (n={n})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
