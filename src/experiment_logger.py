"""Write versioned experiment outputs without overwriting previous runs."""

from __future__ import annotations

import csv
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.config import PROJECT_ROOT


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create_run_dir(results_dir: Path | str, system: str, label: str | None = None) -> Path:
    stamp = utc_timestamp()
    name = f"{stamp}_{label}" if label else stamp
    run_dir = Path(results_dir) / system / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment_metadata(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "packages": package_versions(
            [
                "torch",
                "transformers",
                "datasets",
                "sentence-transformers",
                "faiss-cpu",
                "pillow",
                "numpy",
                "pandas",
            ]
        ),
        "config": dict(config) if config else {},
    }


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key)) for key in fieldnames})


def _stringify(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_manifest(path: Path | str, example_ids: Sequence[str], config: Mapping[str, Any]) -> None:
    write_json(
        path,
        {
            "example_ids": list(example_ids),
            "n_examples": len(example_ids),
            "config": dict(config),
        },
    )
