#!/usr/bin/env python3
"""
Task-level evaluation for Medical VQA (VQA-RAD).

Runs:
1) Inference (optional) via `scripts/gpu/inference_vlm.py` internals
2) Metrics via `src/evaluation/vqa_metrics.py`

Outputs:
- predictions.jsonl
- metrics.json
under `outputs/eval/medical_vqa/` by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
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


def write_subset_jsonl(src: Path, dst: Path, max_samples: int | None) -> Path:
    if not max_samples:
        return src

    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            count += 1
            if count >= max_samples:
                break
    return dst


def run_inference(
    inference_config_path: str, input_file: Path, output_file: Path
) -> dict[str, Any]:
    from scripts.gpu.inference_unified import load_inference_config
    from scripts.gpu.inference_vlm import generate_one, load_model

    defaults = load_inference_config(inference_config_path)
    model = defaults.get("model")
    if not model:
        raise ValueError(f"Missing 'model' in inference config: {inference_config_path}")

    adapter = defaults.get("adapter")
    dtype = str(defaults.get("dtype", "auto"))
    load_4bit = bool(defaults.get("load_4bit", False))
    load_8bit = bool(defaults.get("load_8bit", False))

    max_tokens = int(defaults.get("max_tokens", 64))
    temperature = float(defaults.get("temperature", 0.2))
    top_p = float(defaults.get("top_p", 0.95))

    model_obj, processor = load_model(
        model_name=str(model),
        adapter_path=str(adapter) if adapter else None,
        dtype=dtype,
        load_in_4bit=load_4bit,
        load_in_8bit=load_8bit,
    )

    inputs: list[dict] = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for item in inputs:
            generated = generate_one(
                model=model_obj,
                processor=processor,
                sample=item,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            fout.write(json.dumps({**item, "generated": generated}, ensure_ascii=False) + "\n")

    return {
        "backend": "hf_vlm",
        "model": model,
        "adapter": adapter,
        "inference_config": inference_config_path,
    }


def evaluate_predictions(predictions_file: Path) -> dict[str, Any]:
    from src.evaluation.vqa_metrics import evaluate_vqa

    predictions: list[str] = []
    references: list[str] = []

    with open(predictions_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            predictions.append(str(item.get("generated", "")))
            references.append(str(item.get("answer", "")))

    return evaluate_vqa(predictions=predictions, references=references, compute_yes_no=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Medical VQA (VQA-RAD)")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/evaluation/medical_vqa.yaml",
        help="Evaluation config YAML",
    )
    parser.add_argument(
        "--skip-inference", action="store_true", help="Only compute metrics from predictions.jsonl"
    )
    parser.add_argument(
        "--predictions-file", type=str, default=None, help="Override predictions JSONL path"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = Path(cfg.get("output_dir", "outputs/eval/medical_vqa"))
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_config_path = cfg.get("inference_config")
    if not inference_config_path and not args.skip_inference:
        raise ValueError(
            "Missing 'inference_config' in evaluation config (or pass --skip-inference)."
        )

    test_file = Path(cfg.get("data", {}).get("test_file", "data/processed/medical_vqa/test.jsonl"))
    max_samples = cfg.get("data", {}).get("max_samples")

    predictions_file = (
        Path(args.predictions_file) if args.predictions_file else output_dir / "predictions.jsonl"
    )
    subset_file = output_dir / "test_subset.jsonl"
    eval_input = write_subset_jsonl(test_file, subset_file, max_samples)

    console.print(
        Panel.fit(
            "[bold green]Medical VQA Evaluation[/bold green]\n"
            f"Config: {args.config}\n"
            f"Test file: {test_file}\n"
            f"Output dir: {output_dir}",
            title="eval_medical_vqa",
        )
    )

    inference_meta: dict[str, Any] = {}
    if not args.skip_inference:
        inference_meta = run_inference(
            inference_config_path=str(inference_config_path),
            input_file=eval_input,
            output_file=predictions_file,
        )

    metrics = evaluate_predictions(predictions_file)

    out = {
        "config": cfg,
        "inference": inference_meta,
        "metrics": metrics,
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Wrote metrics: {metrics_path}[/green]")
    console.print(f"[green]Predictions:   {predictions_file}[/green]")


if __name__ == "__main__":
    main()
