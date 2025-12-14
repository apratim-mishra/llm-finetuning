#!/usr/bin/env python3
"""
MLX-VLM batch inference for multimodal datasets (image + question -> answer).

Requires:
  pip install -U mlx-vlm

Usage:
  python scripts/mlx/inference_vlm.py \
    --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
    --input data/processed/medical_vqa/test.jsonl \
    --output outputs/mlx/logs/medical_vqa_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

console = Console()


def _extract_image_and_question(sample: dict) -> tuple[str, str]:
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        image_path = None
        question = None
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image" and part.get("image") and image_path is None:
                        image_path = str(part["image"])
                    if part.get("type") == "text" and part.get("text") and question is None:
                        question = str(part["text"])
            elif isinstance(content, str) and question is None:
                question = content
        if image_path and question:
            return image_path, question

    image_path = str(sample.get("image", "")).strip()
    question = str(sample.get("question", "")).strip()
    if not image_path or not question:
        raise ValueError(
            "Sample missing image/question (expected messages or image/question fields)"
        )
    return image_path, question


def _safe_generate(generate_fn, *args, **kwargs) -> str:
    try:
        return str(generate_fn(*args, **kwargs))
    except TypeError:
        # Fallback if the installed mlx-vlm version doesn't support some kwargs.
        filtered = {k: v for k, v in kwargs.items() if k in {"verbose"}}
        return str(generate_fn(*args, **filtered))


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX-VLM batch inference")
    parser.add_argument(
        "--model", type=str, required=True, help="MLX model repo/id (mlx-community/...)"
    )
    parser.add_argument("--adapter", type=str, default=None, help="Optional adapter path")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input JSONL (with image/question or messages)",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output JSONL (adds generated)"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    try:
        from mlx_vlm import generate, load  # type: ignore
        from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore
        from mlx_vlm.utils import load_config  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mlx-vlm not installed. Install with: pip install -U mlx-vlm") from e

    console.print(f"[blue]Loading MLX-VLM model: {args.model}[/blue]")
    try:
        model, processor = (
            load(args.model, adapter_path=args.adapter) if args.adapter else load(args.model)
        )
    except TypeError:
        model, processor = load(args.model)

    config = load_config(args.model)

    inputs: list[dict[str, Any]] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for item in track(inputs, description="Generating (MLX-VLM)"):
            try:
                image_path, question = _extract_image_and_question(item)
                images = [image_path]
                formatted = apply_chat_template(processor, config, question, num_images=len(images))
                out = _safe_generate(
                    generate,
                    model,
                    processor,
                    formatted,
                    images,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    verbose=False,
                )
                generated = out.strip()
            except Exception:
                generated = ""

            fout.write(json.dumps({**item, "generated": generated}, ensure_ascii=False) + "\n")

    console.print(f"[green]Saved: {out_path}[/green]")


if __name__ == "__main__":
    main()
