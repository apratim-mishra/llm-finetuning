"""
Tests for VQA metrics utilities.
Tests src/evaluation/vqa_metrics.py functions.
"""

import pytest


def test_normalize_vqa_answer_strips_articles_punct_and_whitespace():
    from src.evaluation.vqa_metrics import normalize_vqa_answer

    assert normalize_vqa_answer(" The,   cat! ") == "cat"
    assert normalize_vqa_answer("An MRI.") == "mri"


def test_infer_answer_type_yes_no_vs_open():
    from src.evaluation.vqa_metrics import infer_answer_type

    assert infer_answer_type("Yes") == "yes_no"
    assert infer_answer_type("no.") == "yes_no"
    assert infer_answer_type("maybe") == "open"


def test_exact_match_uses_normalization():
    from src.evaluation.vqa_metrics import exact_match

    assert exact_match("Yes.", "yes") is True
    assert exact_match("a cat", "cat") is True
    assert exact_match("cat", "dog") is False


def test_compute_exact_match_accuracy_raises_on_length_mismatch():
    from src.evaluation.vqa_metrics import compute_exact_match_accuracy

    with pytest.raises(ValueError):
        compute_exact_match_accuracy(["a"], ["a", "b"])


def test_compute_yes_no_accuracy_only_on_yes_no_subset():
    from src.evaluation.vqa_metrics import compute_yes_no_accuracy

    preds = ["yes", "blue", "no"]
    refs = ["Yes", "blue", "no"]

    assert compute_yes_no_accuracy(preds, refs) == 1.0


def test_evaluate_vqa_outputs_expected_fields():
    from src.evaluation.vqa_metrics import evaluate_vqa

    out = evaluate_vqa(predictions=["yes", "no"], references=["yes", "yes"], compute_yes_no=True)
    assert out["num_samples"] == 2
    assert out["num_yes_no"] == 2
    assert out["exact_match"] == 0.5
    assert out["yes_no_accuracy"] == 0.5
