from types import SimpleNamespace

import pytest
from PIL import Image

from src.data_loader import VQAExample
from src.evaluation import compare_runs, find_id_problems
from src.model_loader import extract_generated_text
from src.prompts import build_baseline_prompt, build_conversation, build_rag_prompt, format_evidence
from src.rag_pipeline import retrieve_for_question
from src.embeddings import HashOverlapEncoder
from src.retrieval import Retriever


class FakeProcessor:
    def apply_chat_template(self, messages, add_generation_prompt=True):
        return "TEMPLATE " + messages[0]["content"][1]["text"]

    def __call__(self, text, images, return_tensors="pt"):
        return {"input_ids": FakeTensor([1, 2, 3]), "pixel_values": FakeTensor([0.0])}

    def batch_decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "decoded"):
            return [ids.decoded]
        if isinstance(ids, list):
            return ids
        return [str(ids)]


class FakeTensor:
    def __init__(self, values, decoded=None):
        self.values = values
        self.decoded = decoded
        self.shape = (1, len(values) if hasattr(values, "__len__") else 1)

    def to(self, device):
        return self


class FakeModel:
    def generate(self, **kwargs):
        prompt_len = kwargs["input_ids"].shape[1]
        return FakeGenerated(["ignored", "ignored", "ignored", "yes"], prompt_len=prompt_len)


class FakeGenerated(list):
    def __init__(self, tokens, prompt_len):
        super().__init__(tokens)
        self.shape = (1, len(tokens))
        self.prompt_len = prompt_len
        self.decoded = "yes"

    def __getitem__(self, item):
        if isinstance(item, tuple) and item[0] == slice(None, None, None):
            return FakeGenerated(list(self)[item[1]], prompt_len=0)
        return super().__getitem__(item)


class SliceProcessor(FakeProcessor):
    def batch_decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, FakeGenerated):
            return [" ".join(str(x) for x in ids)]
        return ["yes"]


def test_prompt_requires_question():
    with pytest.raises(ValueError):
        build_baseline_prompt("  ")
    with pytest.raises(ValueError):
        build_rag_prompt("", [])


def test_rag_prompt_separates_evidence_from_image():
    docs = [{"topic": "aorta", "text": "An aneurysm is a ballooning of an artery.", "source_title": "MedlinePlus"}]
    prompt = build_rag_prompt("is there evidence of an aortic aneurysm?", docs)
    assert "not prove" in prompt.lower() or "do not prove" in prompt.lower()
    assert "aortic aneurysm" in prompt
    assert "general" in prompt.lower()


def test_format_evidence_empty():
    assert "No medical passages" in format_evidence([])


def test_conversation_format():
    messages = build_conversation("Question text")
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][1]["text"] == "Question text"


def test_extract_generated_text_uses_new_tokens():
    processor = SliceProcessor()
    generated = FakeGenerated(["a", "b", "c", "yes"], prompt_len=3)
    input_ids = FakeTensor([1, 2, 3])
    text = extract_generated_text(processor, generated, input_ids)
    assert "yes" in text


def test_extract_generated_text_assistant_fallback():
    processor = SimpleNamespace(batch_decode=lambda ids, skip_special_tokens=True: ["User: q Assistant: left"])
    text = extract_generated_text(processor, [[0]], input_ids=None)
    assert text == "left"


def test_invalid_image_rejected():
    from src.baseline import generate_from_prompt

    session = SimpleNamespace(processor=FakeProcessor(), model=FakeModel(), device="cpu")
    with pytest.raises(ValueError, match="PIL"):
        generate_from_prompt(session, "not-an-image", "question")


def test_retrieve_for_question_independent_of_vlm():
    docs = [
        {"id": "d1", "topic": "kidneys", "text": "The kidneys sit in the posterior abdomen beside the spine."},
        {"id": "d2", "topic": "ct", "text": "Computed tomography makes cross sectional x-ray images."},
    ]
    retriever = Retriever(docs, HashOverlapEncoder())
    hits = retrieve_for_question(retriever, "Where are the kidneys?", top_k=1)
    assert hits[0]["id"] == "d1"
    assert "score_l2_distance" in hits[0]


def test_join_missing_and_duplicates():
    baseline = [
        {"example_id": "a", "correct": True, "failed": False, "normalized_prediction": "yes", "normalized_ground_truth": "yes"},
        {"example_id": "b", "correct": False, "failed": False, "normalized_prediction": "no", "normalized_ground_truth": "yes"},
        {"example_id": "b", "correct": False, "failed": False, "normalized_prediction": "no", "normalized_ground_truth": "yes"},
    ]
    rag = [
        {"example_id": "a", "correct": True, "failed": False, "normalized_prediction": "yes", "normalized_ground_truth": "yes"},
        {"example_id": "c", "correct": False, "failed": False, "normalized_prediction": "no", "normalized_ground_truth": "yes"},
    ]
    problems = find_id_problems(baseline)
    assert "b" in problems["duplicated_ids"]
    result = compare_runs(baseline, rag)
    assert result["missing_in_rag"] == ["b"]
    assert result["missing_in_baseline"] == ["c"]
    assert result["n_eligible_pairs"] == 1
    assert result["baseline"]["n_correct"] == 1


def test_failed_inferences_are_not_in_paired_denominator():
    baseline = [
        {"example_id": "a", "correct": True, "failed": False, "normalized_prediction": "yes", "normalized_ground_truth": "yes"},
        {"example_id": "b", "correct": False, "failed": True, "error": "OOM", "normalized_prediction": "", "normalized_ground_truth": "no"},
    ]
    rag = [
        {"example_id": "a", "correct": False, "failed": False, "normalized_prediction": "no", "normalized_ground_truth": "yes"},
        {"example_id": "b", "correct": True, "failed": False, "normalized_prediction": "no", "normalized_ground_truth": "no"},
    ]
    result = compare_runs(baseline, rag)
    assert result["skipped_failed_ids"] == ["b"]
    assert result["n_eligible_pairs"] == 1
    assert result["paired"]["baseline_only_correct"] == 1


def test_run_baseline_with_mocked_model(monkeypatch):
    from src import baseline as baseline_mod
    from src.baseline import run_baseline_example

    image = Image.new("RGB", (8, 8), color="gray")
    example = VQAExample("test-0", "test", 0, "is this an axial image?", "yes", image, {})
    session = SimpleNamespace(
        processor=FakeProcessor(),
        model=FakeModel(),
        device="cpu",
        model_id="mock-vlm",
    )

    def fake_generate(session, image, prompt, max_new_tokens=50):
        return {
            "prompt": prompt,
            "raw_decoded": "Assistant: yes",
            "raw_answer": "yes",
            "inference_seconds": 0.01,
        }

    monkeypatch.setattr(baseline_mod, "generate_from_prompt", fake_generate)
    record = run_baseline_example(session, example)
    assert record["failed"] is False
    assert record["correct"] is True
    assert record["normalized_prediction"] == "yes"
