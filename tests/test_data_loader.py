from PIL import Image

from src.data_loader import VQAExample, _row_to_example


def test_row_to_example_preserves_index_and_rgb():
    image = Image.new("L", (4, 4), color=128)
    row = {"image": image, "question": "is this axial?", "answer": "yes", "qid": "q-9"}
    example = _row_to_example(row, split="test", index=3, require_image=True)
    assert isinstance(example, VQAExample)
    assert example.example_id == "q-9"
    assert example.split == "test"
    assert example.index == 3
    assert example.image.mode == "RGB"


def test_missing_question_raises():
    image = Image.new("RGB", (2, 2), color="white")
    try:
        _row_to_example({"image": image, "question": " ", "answer": "yes"}, "test", 0, True)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "question" in str(exc).lower()
