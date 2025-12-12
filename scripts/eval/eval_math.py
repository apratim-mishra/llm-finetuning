#!/usr/bin/env python3
"""
Task-level evaluation for math reasoning (GSM8K-style).

Runs:
1) Inference (optional) using `scripts/gpu/inference_unified.py` internals
2) Reward/accuracy metrics via `src/rewards/math_reward.py`

Outputs a reproducible `predictions.jsonl` and `metrics.json` under `outputs/eval/...`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()

DEFAULT_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Solve problems step-by-step, showing all your work clearly. "
    'Always end your solution with "Answer: [final numerical answer]".'
)


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    def expand_env(obj: Any) -> Any:
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, list):
            return [expand_env(v) for v in obj]
        if isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        return obj

    return expand_env(cfg)


def write_subset_jsonl(src: Path, dst: Path, max_samples: Optional[int]) -> Path:
    if not max_samples:
        return src

    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= max_samples:
                break
    return dst


def run_inference(
    inference_config_path: str,
    input_file: Path,
    output_file: Path,
    system_prompt_override: Optional[str] = None,
) -> Dict[str, Any]:
    from scripts.gpu.inference_unified import batch_inference, detect_model_family, get_backend, load_inference_config

    defaults = load_inference_config(inference_config_path)
    model = defaults.get("model")
    if not model:
        raise ValueError(f"Missing 'model' in inference config: {inference_config_path}")

    backend_name = defaults.get("backend", "vllm")
    adapter = defaults.get("adapter")

    model_family = defaults.get("model_family") or detect_model_family(model)
    system_prompt = system_prompt_override or defaults.get("system_prompt") or DEFAULT_SYSTEM_PROMPT

    backend = get_backend(backend_name)
    backend.load_model(
        model,
        adapter_path=adapter,
        tensor_parallel_size=int(defaults.get("tensor_parallel", 1)),
        dtype=str(defaults.get("dtype", "auto")),
        max_model_len=defaults.get("max_model_len"),
        gpu_memory_utilization=float(defaults.get("gpu_memory_utilization", 0.9)),
        load_in_4bit=bool(defaults.get("load_4bit", False)),
        load_in_8bit=bool(defaults.get("load_8bit", False)),
    )

    try:
        batch_inference(
            backend=backend,
            input_file=str(input_file),
            output_file=str(output_file),
            model_family=model_family,
            system_prompt=system_prompt,
            max_tokens=int(defaults.get("max_tokens", 512)),
            temperature=float(defaults.get("temperature", 0.1)),
            top_p=float(defaults.get("top_p", 0.95)),
        )
    finally:
        backend.shutdown()

    return {
        "backend": backend_name,
        "model": model,
        "adapter": adapter,
        "model_family": model_family,
        "inference_config": inference_config_path,
    }


def evaluate_predictions(
    predictions_file: Path,
    reward_function: str = "combined",
    include_per_sample: bool = False,
) -> Dict[str, Any]:
    from src.rewards.math_reward import (
        answers_match,
        check_reasoning_quality,
        compute_accuracy,
        extract_answer,
        get_reward_function,
    )

    completions: List[str] = []
    ground_truths: List[str] = []

    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            completions.append(str(item.get("generated", "")))
            if "answer" in item:
                ground_truths.append(str(item.get("answer", "")))
            elif "ground_truth_answer" in item:
                ground_truths.append(str(item.get("ground_truth_answer", "")))
            else:
                ground_truths.append("")

    reward_fn = get_reward_function(reward_function)
    rewards = reward_fn(completions, ground_truths)
    reasoning_scores = [check_reasoning_quality(c) for c in completions]
    accuracy, correct, total = compute_accuracy(completions, ground_truths)

    results: Dict[str, Any] = {
        "reward_function": reward_function,
        "accuracy": round(float(accuracy), 4),
        "correct": int(correct),
        "total": int(total),
        "mean_reward": round(float(sum(rewards) / len(rewards)), 4) if rewards else 0.0,
        "mean_reasoning_score": round(float(sum(reasoning_scores) / len(reasoning_scores)), 4)
        if reasoning_scores
        else 0.0,
    }

    if rewards:
        results["min_reward"] = round(float(min(rewards)), 4)
        results["max_reward"] = round(float(max(rewards)), 4)

    if include_per_sample:
        per_sample = []
        for completion, gt, reward, reasoning in zip(completions, ground_truths, rewards, reasoning_scores):
            pred = extract_answer(completion)
            per_sample.append(
                {
                    "reward": float(reward),
                    "predicted_answer": pred,
                    "ground_truth": gt,
                    "is_correct": bool(answers_match(pred, gt)),
                    "reasoning_score": float(reasoning),
                }
            )
        results["results"] = per_sample

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate math task (inference + reward metrics)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/evaluation/math.yaml",
        help="Evaluation config YAML",
    )
    parser.add_argument("--skip-inference", action="store_true", help="Only compute metrics from predictions.jsonl")
    parser.add_argument("--predictions-file", type=str, default=None, help="Override predictions JSONL path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = Path(cfg.get("output_dir", "outputs/eval/math"))
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_config_path = cfg.get("inference_config")
    if not inference_config_path and not args.skip_inference:
        raise ValueError("Missing 'inference_config' in evaluation config (or pass --skip-inference).")

    test_file = Path(cfg.get("data", {}).get("test_file", "data/processed/math/gsm8k_test.jsonl"))
    max_samples = cfg.get("data", {}).get("max_samples")

    predictions_file = Path(args.predictions_file) if args.predictions_file else output_dir / "predictions.jsonl"
    subset_file = output_dir / "test_subset.jsonl"
    eval_input = write_subset_jsonl(test_file, subset_file, max_samples)

    console.print(
        Panel.fit(
            "[bold green]Math Evaluation[/bold green]\n"
            f"Config: {args.config}\n"
            f"Test file: {test_file}\n"
            f"Output dir: {output_dir}",
            title="eval_math",
        )
    )

    inference_meta: Dict[str, Any] = {}
    if not args.skip_inference:
        inference_meta = run_inference(
            inference_config_path=str(inference_config_path),
            input_file=eval_input,
            output_file=predictions_file,
        )

    reward_fn = cfg.get("reward_function", "combined")
    include_per_sample = bool(cfg.get("include_per_sample", False))
    results = evaluate_predictions(
        predictions_file,
        reward_function=reward_fn,
        include_per_sample=include_per_sample,
    )

    out = {
        "config": cfg,
        "inference": inference_meta,
        "metrics": results,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Wrote metrics: {metrics_path}[/green]")
    console.print(f"[green]Predictions:   {predictions_file}[/green]")


if __name__ == "__main__":
    main()
