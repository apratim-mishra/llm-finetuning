#!/usr/bin/env python3
"""
Export a (merged) Hugging Face model to ONNX (best-effort).

Notes:
- Decoder-only LLM ONNX export can be large and may require additional tuning depending on model family.
- Recommended workflow is: train LoRA -> merge (`scripts/gpu/merge_lora.py`) -> export ONNX.

This script uses `python -m transformers.onnx` if available.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()


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


def run_transformers_onnx_export(
    model: str,
    output_dir: Path,
    feature: str = "causal-lm",
    opset: int = 17,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "transformers.onnx",
        "--model",
        model,
        "--feature",
        feature,
        "--opset",
        str(opset),
        str(output_dir),
    ]

    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HF model to ONNX (best-effort)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Optional YAML config (configs/export/*.yaml)")
    parser.add_argument("--model", type=str, default=None, help="HF model id/path (prefer merged model dir)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for ONNX files")
    parser.add_argument("--feature", type=str, default=None, help='Transformers ONNX feature (e.g., "causal-lm")')
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset (default 17)")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)

    model = args.model or cfg.get("model")
    output_dir = args.output_dir or cfg.get("output_dir")
    feature = args.feature or cfg.get("feature", "causal-lm")
    opset = int(args.opset or cfg.get("opset", 17))

    if not model or not output_dir:
        console.print("[red]Missing --model and/or --output-dir (or provide --config)[/red]")
        raise SystemExit(2)

    console.print(
        Panel.fit(
            "[bold green]ONNX Export[/bold green]\n"
            f"Model: {model}\n"
            f"Output: {output_dir}\n"
            f"Feature: {feature}\n"
            f"Opset: {opset}",
            title="export_onnx",
        )
    )

    try:
        run_transformers_onnx_export(model=model, output_dir=Path(output_dir), feature=feature, opset=opset)
    except subprocess.CalledProcessError as e:
        console.print("[red]ONNX export failed.[/red]")
        console.print(
            "[yellow]Tip: ONNX export for decoder-only LLMs can require extra setup. "
            "Start from a merged model directory (no adapters) and ensure `transformers` is installed.[/yellow]"
        )
        raise SystemExit(e.returncode) from e

    console.print("[bold green]Done.[/bold green]")


if __name__ == "__main__":
    main()

