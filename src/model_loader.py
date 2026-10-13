"""Load the vision-language model once per process and detect devices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

_SESSION: "VLMSession | None" = None


@dataclass
class VLMSession:
    model_id: str
    processor: Any
    model: Any
    device: str
    torch_dtype: str


def detect_device(preference: str = "auto") -> str:
    """Choose MPS, CUDA, or CPU. Preference may force a specific device."""
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return "cpu"
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available on this machine.")
        return "mps"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return "cuda"
    if preference != "auto":
        raise ValueError(f"Unknown device preference: {preference}")
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def load_vlm(
    model_id: str,
    device_preference: str = "auto",
    torch_dtype: str = "float16",
    force_reload: bool = False,
) -> VLMSession:
    """Load processor and model, caching them in this Python process."""
    global _SESSION
    device = detect_device(device_preference)
    if (
        _SESSION is not None
        and not force_reload
        and _SESSION.model_id == model_id
        and _SESSION.device == device
        and _SESSION.torch_dtype == torch_dtype
    ):
        return _SESSION

    processor = AutoProcessor.from_pretrained(model_id)
    dtype = _dtype_from_name(torch_dtype)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    _SESSION = VLMSession(
        model_id=model_id,
        processor=processor,
        model=model,
        device=device,
        torch_dtype=torch_dtype,
    )
    return _SESSION


def move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Move tensor values in a processor batch onto the target device."""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def extract_generated_text(
    processor: Any,
    generated_ids: Any,
    input_ids: Any | None = None,
) -> str:
    """Decode newly generated tokens.

    Prefer slicing after the prompt length. If that is unavailable, decode
    the full sequence and take text after the last ``Assistant:`` marker.
    Do not assume capitalization of the rest of the string.
    """
    if input_ids is not None and hasattr(generated_ids, "shape") and hasattr(input_ids, "shape"):
        prompt_len = int(input_ids.shape[-1])
        new_tokens = generated_ids[:, prompt_len:] if generated_ids.shape[-1] >= prompt_len else generated_ids
        decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        text = str(decoded).strip()
        if text:
            return text

    decoded_full = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    text = str(decoded_full).strip()
    marker = "Assistant:"
    if marker in text:
        return text.split(marker)[-1].strip()
    lower_marker = "assistant:"
    lowered = text.lower()
    if lower_marker in lowered:
        idx = lowered.rfind(lower_marker)
        return text[idx + len(lower_marker) :].strip()
    return text
