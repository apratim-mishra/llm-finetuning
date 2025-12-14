"""
VQA-style evaluation utilities.

Currently used for Medical VQA (VQA-RAD) where answers are short strings (often yes/no).
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans(dict.fromkeys(string.punctuation, " "))


def normalize_vqa_answer(text: str) -> str:
    """
    Normalize answers for robust string matching:
    - lowercase
    - remove punctuation
    - remove English articles
    - collapse whitespace
    """
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = s.translate(_PUNCT_TABLE)
    tokens = [t for t in s.split() if t and t not in _ARTICLES]
    return " ".join(tokens)


def infer_answer_type(answer: str) -> str:
    ans = normalize_vqa_answer(answer)
    if ans in {"yes", "no"}:
        return "yes_no"
    return "open"


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_vqa_answer(prediction) == normalize_vqa_answer(reference)


def compute_exact_match_accuracy(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    if not predictions:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references, strict=True) if exact_match(p, r))
    return correct / len(predictions)


def compute_yes_no_accuracy(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    idx = [i for i, r in enumerate(references) if infer_answer_type(r) == "yes_no"]
    if not idx:
        return 0.0

    correct = 0
    for i in idx:
        pred = normalize_vqa_answer(predictions[i])
        ref = normalize_vqa_answer(references[i])
        if pred in {"yes", "no"} and pred == ref:
            correct += 1
    return correct / len(idx)


@dataclass(frozen=True)
class VqaMetrics:
    exact_match: float
    yes_no_accuracy: float | None = None
    num_samples: int = 0
    num_yes_no: int = 0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "exact_match": float(self.exact_match),
            "num_samples": int(self.num_samples),
            "num_yes_no": int(self.num_yes_no),
        }
        if self.yes_no_accuracy is not None:
            out["yes_no_accuracy"] = float(self.yes_no_accuracy)
        return out


def evaluate_vqa(
    predictions: list[str],
    references: list[str],
    compute_yes_no: bool = True,
) -> dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")

    em = compute_exact_match_accuracy(predictions, references)

    yes_no_acc = None
    num_yes_no = 0
    if compute_yes_no:
        yes_no_refs = [r for r in references if infer_answer_type(r) == "yes_no"]
        num_yes_no = len(yes_no_refs)
        yes_no_acc = compute_yes_no_accuracy(predictions, references)

    metrics = VqaMetrics(
        exact_match=round(float(em), 4),
        yes_no_accuracy=round(float(yes_no_acc), 4) if yes_no_acc is not None else None,
        num_samples=len(predictions),
        num_yes_no=num_yes_no,
    )
    return metrics.to_dict()
