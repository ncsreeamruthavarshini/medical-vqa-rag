"""Project configuration and path helpers.

Settings live in ``config/config.yaml`` so experiments can be reproduced
without editing Python files. Paths are resolved from the project root,
not from whoever happens to be the current working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration and attach resolved filesystem paths."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    paths = data.setdefault("paths", {})
    data["_config_path"] = str(config_path)
    data["_project_root"] = str(PROJECT_ROOT)
    data["_resolved_paths"] = {
        "knowledge_base": _resolve(paths.get("knowledge_base", "data/knowledge_base/medical_knowledge.json")),
        "retrieval_eval_queries": _resolve(
            paths.get("retrieval_eval_queries", "data/knowledge_base/retrieval_eval_queries.json")
        ),
        "results_dir": _resolve(paths.get("results_dir", "results")),
        "processed_dir": _resolve(paths.get("processed_dir", "data/processed")),
    }
    return data


def _resolve(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_device_preference(config: dict[str, Any] | None = None) -> str:
    """Return the configured device string: auto, mps, cuda, or cpu."""
    config = config or load_config()
    return str(config.get("model", {}).get("device", "auto"))
