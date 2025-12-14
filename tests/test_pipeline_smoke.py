"""
End-to-end-ish smoke tests (CPU-only).

These tests validate that the new "next steps" utilities wire together without requiring
GPUs, model downloads, or network access.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_manifest_sha256(temp_dir: Path):
    import hashlib

    from src.data.manifest import sha256_file

    path = temp_dir / "sample.txt"
    path.write_text("hello\n", encoding="utf-8")

    expected = hashlib.sha256(b"hello\n").hexdigest()
    assert sha256_file(path) == expected


def test_inference_config_flattening(temp_dir: Path):
    from scripts.gpu.inference_unified import load_inference_config

    cfg_path = temp_dir / "inference.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "backend: vllm",
                "model: Qwen/Qwen2.5-7B-Instruct",
                "generation:",
                "  max_tokens: 123",
                "  temperature: 0.4",
                "runtime:",
                "  tensor_parallel: 2",
                "  max_model_len: 4096",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_inference_config(str(cfg_path))
    assert cfg["backend"] == "vllm"
    assert cfg["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg["max_tokens"] == 123
    assert cfg["temperature"] == 0.4
    assert cfg["tensor_parallel"] == 2
    assert cfg["max_model_len"] == 4096


def test_training_config_profile_merge(temp_dir: Path):
    from src.training.config_utils import load_config_with_profile

    base = temp_dir / "base.yaml"
    profile = temp_dir / "profile.yaml"

    base.write_text(
        "\n".join(
            [
                "model:",
                "  name: Qwen/Qwen2.5-7B-Instruct",
                "training:",
                "  bf16: true",
                "  per_device_train_batch_size: 4",
            ]
        ),
        encoding="utf-8",
    )
    profile.write_text(
        "\n".join(
            [
                "training:",
                "  bf16: false",
                "  per_device_train_batch_size: 1",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config_with_profile(str(base), profile_path=str(profile))
    assert cfg["training"]["bf16"] is False
    assert cfg["training"]["per_device_train_batch_size"] == 1


def test_preference_pair_selection_binary_reward():
    from src.data.preferences import select_preference_pair
    from src.rewards.math_reward import get_reward_function

    reward_base = get_reward_function("binary")
    reward_fn = lambda comps, gts: reward_base(comps, gts)  # noqa: E731

    pair = select_preference_pair(
        prompt="Solve: 2+2",
        candidates=["Answer: 4", "Answer: 5"],
        ground_truth="4",
        reward_fn=reward_fn,
    )

    assert pair is not None
    assert "4" in pair.chosen
    assert "5" in pair.rejected
    assert pair.chosen_reward > pair.rejected_reward


def test_eval_translation_metrics_no_bleu(temp_dir: Path, sample_translation_messages: list[dict]):
    from scripts.eval.eval_translation import evaluate_predictions

    predictions_path = temp_dir / "preds.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for item in sample_translation_messages:
            f.write(json.dumps({**item, "generated": "Hello"}, ensure_ascii=False) + "\n")

    results = evaluate_predictions(predictions_path, metrics=[])
    assert results["num_samples"] == len(sample_translation_messages)
    assert "length_ratio_mean" in results


def test_eval_math_metrics_binary_reward(temp_dir: Path, sample_math_jsonl: Path):
    from scripts.eval.eval_math import evaluate_predictions

    predictions_path = temp_dir / "math_preds.jsonl"
    with (
        open(sample_math_jsonl, encoding="utf-8") as fin,
        open(predictions_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            item = json.loads(line)
            fout.write(json.dumps({**item, "generated": f"Answer: {item['answer']}"}) + "\n")

    results = evaluate_predictions(
        predictions_path, reward_function="binary", include_per_sample=True
    )
    assert results["accuracy"] == 1.0
    assert results["correct"] == results["total"]
    assert results["max_reward"] == 1.0
    assert len(results["results"]) == results["total"]


def test_eval_medical_vqa_metrics(temp_dir: Path):
    from scripts.eval.eval_medical_vqa import evaluate_predictions

    predictions_path = temp_dir / "vqa_preds.jsonl"
    rows = [
        {"id": "1", "answer": "yes", "generated": "Yes."},
        {"id": "2", "answer": "no", "generated": "maybe"},
        {"id": "3", "answer": "left lung", "generated": "left lung"},
    ]

    with open(predictions_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    results = evaluate_predictions(predictions_path)
    assert results["num_samples"] == 3
    assert results["num_yes_no"] == 2
    assert results["exact_match"] == 0.6667
    assert results["yes_no_accuracy"] == 0.5


def test_generate_math_preferences_offline_candidates(temp_dir: Path):
    input_path = temp_dir / "math_candidates.jsonl"
    output_dir = temp_dir / "out"

    rows = [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "candidates": ["Let's solve. Answer: 4", "Answer: 5"],
        },
        {
            "question": "What is 10-3?",
            "answer": "7",
            "candidates": ["Answer: 6", "Answer: 7"],
        },
    ]

    with open(input_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    cmd = [
        sys.executable,
        "data/scripts/generate_math_preferences.py",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--backend",
        "hf",
        "--reward-function",
        "binary",
        "--val-ratio",
        "0.5",
    ]
    subprocess.run(cmd, check=True)

    train_out = output_dir / "preference_pairs.jsonl"
    val_out = output_dir / "preference_pairs_val.jsonl"
    assert train_out.exists()
    assert val_out.exists()
