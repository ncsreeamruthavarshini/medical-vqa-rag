from src.metrics import exact_match_accuracy, paired_outcomes, recall_at_k, yes_no_accuracy


def test_exact_match_hand_checked():
    preds = ["Yes", "left", "the liver"]
    golds = ["yes", "left", "diaphragm"]
    result = exact_match_accuracy(preds, golds)
    assert result["n"] == 3
    assert result["correct"] == 2
    assert result["incorrect"] == 1
    assert abs(result["exact_match_accuracy"] - (2 / 3)) < 1e-9


def test_yes_no_subset():
    preds = ["yes", "nope", "left"]
    golds = ["yes", "no", "left"]
    result = yes_no_accuracy(preds, golds)
    assert result["n_yes_no"] == 2
    assert result["correct"] == 1


def test_recall_at_k():
    retrieved = [["a", "b", "c"], ["x", "y"], ["p"]]
    relevant = [["b", "z"], ["x"], []]
    result = recall_at_k(retrieved, relevant, k=2)
    assert result["n_labeled_queries"] == 2
    assert result["n_skipped_unlabeled"] == 1
    # query 0: 1 of 2 relevant in top-2; query 1: 1 of 1
    assert abs(result["recall_at_k"] - 0.75) < 1e-9


def test_paired_outcomes():
    result = paired_outcomes([True, True, False, False], [True, False, True, False])
    assert result == {
        "n_paired": 4,
        "both_correct": 1,
        "baseline_only_correct": 1,
        "rag_only_correct": 1,
        "both_incorrect": 1,
    }
