#!/usr/bin/env python3
"""Compare versioned baseline and RAG prediction CSVs on the same example ids."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation import compare_runs, format_comparison_report, load_prediction_csv, save_comparison_outputs
from src.experiment_logger import create_run_dir, write_json


def latest_predictions(results_dir: Path, system: str) -> Path:
    root = results_dir / system
    if not root.exists():
        raise FileNotFoundError(f"No {system} results directory at {root}")
    candidates = sorted(root.glob("*/predictions.csv"))
    if not candidates:
        raise FileNotFoundError(f"No predictions.csv files under {root}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and RAG result files")
    parser.add_argument("--baseline", default=None, help="Path to baseline predictions.csv")
    parser.add_argument("--rag", default=None, help="Path to RAG predictions.csv")
    parser.add_argument("--config", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config["_resolved_paths"]["results_dir"]
    baseline_path = Path(args.baseline) if args.baseline else latest_predictions(results_dir, "baseline")
    rag_path = Path(args.rag) if args.rag else latest_predictions(results_dir, "rag")

    print(f"Baseline file: {baseline_path}")
    print(f"RAG file:      {rag_path}")

    baseline_rows = load_prediction_csv(baseline_path)
    rag_rows = load_prediction_csv(rag_path)
    result = compare_runs(baseline_rows, rag_rows)
    print(format_comparison_report(result))

    output_dir = create_run_dir(results_dir, "comparison", label=args.label)
    save_comparison_outputs(result, output_dir)
    write_json(
        output_dir / "inputs.json",
        {"baseline": str(baseline_path), "rag": str(rag_path)},
    )
    print(f"Saved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
