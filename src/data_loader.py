"""Load and validate VQA-RAD examples without changing split order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image


@dataclass
class VQAExample:
    example_id: str
    split: str
    index: int
    question: str
    answer: str
    image: Image.Image | None
    extra: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("image", None)
        return data


def inspect_dataset(dataset_id: str) -> dict[str, Any]:
    """Return split names and column names without iterating all rows."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id)
    splits = {}
    for name, split in dataset.items():
        splits[name] = {
            "num_rows": len(split),
            "columns": list(split.column_names),
        }
    return {"dataset_id": dataset_id, "splits": splits}


def load_vqa_examples(
    dataset_id: str,
    split: str,
    n_examples: int | None = None,
    start_index: int = 0,
    require_image: bool = True,
) -> list[VQAExample]:
    """Load a contiguous slice of a split, preserving original order.

    ``n_examples=None`` loads from ``start_index`` to the end of the split.
    Test answers are returned for evaluation only. Callers must not write
    them into the knowledge base.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_id)
    if split not in dataset:
        available = ", ".join(dataset.keys())
        raise ValueError(f"Split '{split}' not found. Available: {available}")

    rows = dataset[split]
    total = len(rows)
    if start_index < 0 or start_index >= total:
        raise ValueError(f"start_index {start_index} is outside split length {total}")

    end_index = total if n_examples is None else min(total, start_index + n_examples)
    examples: list[VQAExample] = []
    for index in range(start_index, end_index):
        row = rows[index]
        example = _row_to_example(row, split=split, index=index, require_image=require_image)
        examples.append(example)
    return examples


def _row_to_example(row: dict[str, Any], split: str, index: int, require_image: bool) -> VQAExample:
    question = row.get("question")
    answer = row.get("answer")
    if question is None or str(question).strip() == "":
        raise ValueError(f"Missing question at {split}[{index}]")
    if answer is None:
        raise ValueError(f"Missing answer at {split}[{index}]")

    image = row.get("image")
    if image is not None and not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image) if hasattr(image, "shape") else Image.open(image)
        except Exception as exc:
            if require_image:
                raise ValueError(f"Invalid image at {split}[{index]}") from exc
            image = None
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    elif require_image:
        raise ValueError(f"Missing image at {split}[{index}]")

    known = {"image", "question", "answer"}
    extra = {key: value for key, value in row.items() if key not in known and not _is_image_like(value)}
    example_id = str(row.get("qid") or row.get("id") or f"{split}-{index}")
    return VQAExample(
        example_id=example_id,
        split=split,
        index=index,
        question=str(question),
        answer=str(answer),
        image=image,
        extra=extra,
    )


def _is_image_like(value: Any) -> bool:
    return isinstance(value, Image.Image)
