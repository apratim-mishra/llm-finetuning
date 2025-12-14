#!/usr/bin/env python3
"""
Medical VQA (VQA-RAD) Data Preparation (Use Case 3)

Downloads and processes the VQA-RAD dataset into a reusable JSONL format:
- Saves (deduplicated) images to `data/processed/medical_vqa/images/`
- Writes train/val/test JSONL files with both simple fields and multimodal `messages`

Dataset: https://huggingface.co/datasets/flaviagiammarino/vqa-rad
"""

from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()


# =============================================================================
# Configuration
# =============================================================================

DATA_CONFIG: dict[str, Any] = {
    "output_dir": "data/processed/medical_vqa",
    "val_ratio": 0.1,
    "seed": 42,
    "max_train_samples": None,  # Set for quick dev runs (e.g., 500)
}

SYSTEM_PROMPT = (
    "You are a medical visual question answering assistant. "
    "Answer questions about the given radiology image accurately and concisely. "
    "If the question is yes/no, respond with exactly 'yes' or 'no'."
)


# =============================================================================
# Utilities
# =============================================================================


def _normalize_answer(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    # Common cleanup for yes/no
    if cleaned in {"y", "yeah", "yep", "true"}:
        return "yes"
    if cleaned in {"n", "nope", "false"}:
        return "no"
    return cleaned


def _infer_answer_type(answer: str) -> str:
    ans = _normalize_answer(answer)
    if ans in {"yes", "no"}:
        return "yes_no"
    return "open"


def _save_image_dedup(image, images_dir: Path) -> str:
    """
    Save a PIL image (or datasets Image) to disk with content-based naming to deduplicate.
    Returns a workspace-relative path string.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    # `datasets` returns a PIL.Image.Image for Image features
    buf = BytesIO()
    image.save(buf, format="PNG")  # deterministic enough for dedup within run
    png_bytes = buf.getvalue()
    digest = sha256(png_bytes).hexdigest()
    filename = f"{digest[:16]}.png"
    out_path = images_dir / filename
    if not out_path.exists():
        out_path.write_bytes(png_bytes)
    return str(out_path)


def _format_multimodal_messages(
    image_path: str, question: str, answer: str
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": answer},
    ]


def _save_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =============================================================================
# Main pipeline
# =============================================================================


def prepare_medical_vqa_data(use_full: bool = True) -> None:
    console.print("[bold green]═══ Medical VQA (VQA-RAD) Data Preparation ═══[/bold green]")

    output_dir = Path(DATA_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"

    console.print("[bold blue]Loading dataset: flaviagiammarino/vqa-rad[/bold blue]")
    ds = load_dataset("flaviagiammarino/vqa-rad")

    if "train" not in ds or "test" not in ds:
        raise ValueError("Expected 'train' and 'test' splits in flaviagiammarino/vqa-rad")

    train_split = ds["train"]
    test_split = ds["test"]

    # Optional downsample for quick dev runs
    max_train = DATA_CONFIG.get("max_train_samples")
    if max_train and not use_full:
        train_split = train_split.shuffle(seed=DATA_CONFIG["seed"]).select(
            range(min(int(max_train), len(train_split)))
        )

    # Create train/val split from train
    val_ratio = float(DATA_CONFIG.get("val_ratio", 0.1))
    split = train_split.train_test_split(test_size=val_ratio, seed=int(DATA_CONFIG["seed"]))
    train_rows = split["train"]
    val_rows = split["test"]

    console.print(
        f"[cyan]Split sizes:[/cyan] train={len(train_rows)} val={len(val_rows)} test={len(test_split)}"
    )

    def to_row(item: dict[str, Any], split_name: str, idx: int) -> dict[str, Any]:
        image_path = _save_image_dedup(item["image"], images_dir=images_dir)
        question = str(item.get("question", "")).strip()
        answer = _normalize_answer(item.get("answer", ""))

        return {
            "id": f"vqa_rad_{split_name}_{idx}",
            "image": image_path,
            "question": question,
            "answer": answer,
            "answer_type": _infer_answer_type(answer),
            "messages": _format_multimodal_messages(image_path, question, answer),
            "source": "vqa-rad",
        }

    console.print("[bold blue]Writing JSONL + images...[/bold blue]")
    train_out: list[dict[str, Any]] = []
    val_out: list[dict[str, Any]] = []
    test_out: list[dict[str, Any]] = []

    for i, item in enumerate(track(train_rows, description="Processing train")):
        train_out.append(to_row(item, "train", i))
    for i, item in enumerate(track(val_rows, description="Processing val")):
        val_out.append(to_row(item, "val", i))
    for i, item in enumerate(track(test_split, description="Processing test")):
        test_out.append(to_row(item, "test", i))

    _save_jsonl(train_out, output_dir / "train.jsonl")
    _save_jsonl(val_out, output_dir / "val.jsonl")
    _save_jsonl(test_out, output_dir / "test.jsonl")

    # Save config for reproducibility
    config_path = output_dir / "data_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                **DATA_CONFIG,
                "system_prompt": SYSTEM_PROMPT,
                "actual_train": len(train_out),
                "actual_val": len(val_out),
                "actual_test": len(test_out),
            },
            f,
        )

    # Write manifest (hashes + dataset stats)
    try:
        from src.data.manifest import build_files_manifest, write_manifest

        def _count_types(items: list[dict[str, Any]]) -> dict[str, int]:
            return dict(Counter(s.get("answer_type", "unknown") for s in items))

        manifest = {
            "task": "medical_vqa_vqa_rad",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform": {
                "python": sys.version.split()[0],
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "config": {
                **DATA_CONFIG,
                "use_full": bool(use_full),
            },
            "counts": {
                "train": len(train_out),
                "val": len(val_out),
                "test": len(test_out),
                "unique_images": len(list(images_dir.glob("*.png"))),
            },
            "answer_types": {
                "train": _count_types(train_out),
                "val": _count_types(val_out),
                "test": _count_types(test_out),
            },
            "files": build_files_manifest(
                {
                    "train.jsonl": output_dir / "train.jsonl",
                    "val.jsonl": output_dir / "val.jsonl",
                    "test.jsonl": output_dir / "test.jsonl",
                    "data_config.yaml": config_path,
                }
            ),
        }

        write_manifest(output_dir / "manifest.json", manifest)
        console.print(f"[green]Wrote manifest: {output_dir / 'manifest.json'}[/green]")
    except Exception as e:  # pragma: no cover
        console.print(f"[yellow]Warning: could not write manifest.json: {e}[/yellow]")

    console.print("\n[bold green]═══ Data Preparation Complete ═══[/bold green]")
    console.print(f"  Output dir:  {output_dir}")
    console.print(f"  Images dir:  {images_dir}")
    console.print(f"  Train/Val/Test: {len(train_out)}/{len(val_out)}/{len(test_out)}")

    if train_out:
        console.print("\n[bold]Sample entry:[/bold]")
        console.print(json.dumps(train_out[0], indent=2, ensure_ascii=False)[:800] + "...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare VQA-RAD medical VQA data")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full dataset (default). If omitted, may downsample using DATA_CONFIG.max_train_samples.",
    )
    args = parser.parse_args()

    prepare_medical_vqa_data(
        use_full=bool(args.full) or DATA_CONFIG.get("max_train_samples") is None
    )
