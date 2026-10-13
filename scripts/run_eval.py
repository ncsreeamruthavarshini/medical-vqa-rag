"""Shared helpers for evaluation scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_loader import load_vqa_examples
from src.experiment_logger import create_run_dir, environment_metadata, write_csv, write_json, write_manifest
from src.metrics import summarize_records


def parse_eval_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--n-examples", type=int, default=None, help="Override number of examples")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--label", default=None, help="Optional folder label")
    parser.add_argument("--smoke", action="store_true", help="Use smoke_n_examples from config")
    return parser.parse_args()


def resolve_eval_settings(args: argparse.Namespace) -> dict:
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    n_examples = args.n_examples
    if n_examples is None:
        n_examples = dataset_cfg["smoke_n_examples"] if args.smoke else dataset_cfg["n_examples"]
    return {
        "config": config,
        "dataset_id": dataset_cfg["id"],
        "split": args.split or dataset_cfg["eval_split"],
        "n_examples": n_examples,
        "start_index": args.start_index if args.start_index is not None else dataset_cfg["start_index"],
        "max_new_tokens": args.max_new_tokens or config["generation"]["max_new_tokens"],
        "top_k": args.top_k or config["retrieval"]["top_k"],
        "label": args.label,
    }


def load_eval_examples(settings: dict):
    return load_vqa_examples(
        dataset_id=settings["dataset_id"],
        split=settings["split"],
        n_examples=settings["n_examples"],
        start_index=settings["start_index"],
    )


def save_run(system: str, settings: dict, records: list[dict], extra: dict | None = None) -> Path:
    config = settings["config"]
    results_dir = config["_resolved_paths"]["results_dir"]
    run_dir = create_run_dir(results_dir, system, label=settings.get("label"))
    summary = summarize_records(records)
    payload = {
        "system": system,
        "summary": summary,
        "settings": {
            "dataset_id": settings["dataset_id"],
            "split": settings["split"],
            "n_examples_requested": settings["n_examples"],
            "start_index": settings["start_index"],
            "max_new_tokens": settings["max_new_tokens"],
            "top_k": settings.get("top_k"),
            "example_ids": [row["example_id"] for row in records],
        },
        "environment": environment_metadata(
            {
                "dataset": config.get("dataset"),
                "model": config.get("model"),
                "generation": config.get("generation"),
                "retrieval": config.get("retrieval"),
            }
        ),
        "extra": extra or {},
    }
    write_csv(run_dir / "predictions.csv", records)
    write_json(run_dir / "summary.json", payload)
    write_manifest(
        run_dir / "manifest.json",
        [row["example_id"] for row in records],
        payload["settings"],
    )
    print(f"Saved {system} results to {run_dir}")
    print(
        f"Evaluated: {summary['n_evaluated']} | Failed: {summary['n_failed']} | "
        f"Correct: {summary['n_correct']} | "
        f"Accuracy: {summary['exact_match_accuracy']}"
    )
    return run_dir
