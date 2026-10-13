#!/usr/bin/env python3
"""Run the baseline Medical VQA system on a configurable evaluation slice."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_eval import load_eval_examples, parse_eval_args, resolve_eval_settings, save_run
from src.baseline import run_baseline_example
from src.model_loader import load_vlm


def main() -> None:
    args = parse_eval_args("Run baseline Medical VQA")
    settings = resolve_eval_settings(args)
    config = settings["config"]
    print("Loading evaluation examples...")
    examples = load_eval_examples(settings)
    print(f"Loaded {len(examples)} examples from {settings['split']}[{settings['start_index']}:]")

    print("Loading vision-language model (this can take several minutes)...")
    session = load_vlm(
        model_id=config["model"]["vlm_id"],
        device_preference=config["model"]["device"],
        torch_dtype=config["model"]["torch_dtype"],
    )
    print(f"Device: {session.device}")

    records = []
    for i, example in enumerate(examples, start=1):
        print(f"[{i}/{len(examples)}] {example.example_id}: {example.question}")
        record = run_baseline_example(
            session,
            example,
            max_new_tokens=settings["max_new_tokens"],
        )
        records.append(record)
        status = "FAIL" if record["failed"] else ("OK" if record["correct"] else "wrong")
        print(f"  -> {record.get('raw_answer', '')!r} [{status}]")

    save_run("baseline", settings, records)


if __name__ == "__main__":
    main()
