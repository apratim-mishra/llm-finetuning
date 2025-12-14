#!/usr/bin/env python3
"""
MLX Vision-Language LoRA Training Script (Mac Apple Silicon)

Wraps `mlx_vlm.lora` to fine-tune a VLM on image+text datasets (e.g., VQA-RAD).

Usage:
  python scripts/mlx/train_vlm_sft.py --config configs/mlx/vlm_medical_vqa.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    def expand_env(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [expand_env(v) for v in obj]
        return obj

    return expand_env(cfg)


def _validate_jsonl(path: Path, max_samples: int = 3) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    ok = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            json.loads(line)
            ok += 1
            if ok >= max_samples:
                break
    if ok == 0:
        raise ValueError(f"Empty JSONL: {path}")


def prepare_mlx_data(train_file: str, val_file: str | None, out_dir: Path) -> Path:
    data_dir = out_dir / "mlx_vlm_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_src = Path(train_file)
    val_src = Path(val_file) if val_file else None

    _validate_jsonl(train_src)
    if val_src:
        _validate_jsonl(val_src)

    train_dst = data_dir / "train.jsonl"
    shutil.copy(train_src, train_dst)

    if val_src and val_src.exists():
        val_dst = data_dir / "valid.jsonl"
        shutil.copy(val_src, val_dst)

    return data_dir


def build_mlx_vlm_command(config: dict, data_dir: Path) -> list[str]:
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    ckpt_cfg = config.get("checkpoint", {})

    cmd = [
        sys.executable,
        "-m",
        "mlx_vlm.lora",
        "--model",
        model_cfg.get("name", "mlx-community/Qwen2-VL-2B-Instruct-4bit"),
        "--data",
        str(data_dir),
        "--train",
        "--iters",
        str(train_cfg.get("iters", 500)),
        "--batch-size",
        str(train_cfg.get("batch_size", 1)),
        "--lora-layers",
        str(lora_cfg.get("layers", 16)),
        "--lora-rank",
        str(lora_cfg.get("rank", 16)),
        "--learning-rate",
        str(train_cfg.get("learning_rate", 1e-4)),
        "--save-every",
        str(ckpt_cfg.get("save_every", 100)),
        "--adapter-path",
        ckpt_cfg.get("output_dir", "outputs/mlx/adapters/medical_vqa_vlm"),
    ]

    if model_cfg.get("max_seq_length"):
        cmd.extend(["--max-seq-length", str(model_cfg["max_seq_length"])])

    if train_cfg.get("seed") is not None:
        cmd.extend(["--seed", str(train_cfg["seed"])])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX VLM LoRA training wrapper")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to MLX VLM YAML config"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_file = data_cfg.get("train_file")
    val_file = data_cfg.get("val_file")
    if not train_file:
        raise ValueError("Missing data.train_file in config")

    output_dir = Path(
        cfg.get("checkpoint", {}).get("output_dir", "outputs/mlx/adapters/medical_vqa_vlm")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            "[bold green]MLX VLM LoRA Training[/bold green]\n"
            f"Config: {args.config}\n"
            f"Model:  {cfg.get('model', {}).get('name')}\n"
            f"Data:   {train_file}",
            title="mlx_vlm.lora",
        )
    )

    data_dir = prepare_mlx_data(train_file=train_file, val_file=val_file, out_dir=output_dir)
    cmd = build_mlx_vlm_command(cfg, data_dir=data_dir)

    console.print("[bold]Running:[/bold] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
