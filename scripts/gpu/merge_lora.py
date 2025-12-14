#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into a base model and write a standalone HF model directory.

This is useful for:
- exporting a single deployable artifact (no adapter dependency)
- preparing models for ONNX/TensorRT-style toolchains
- merging VLM adapters when the base model is multimodal (best-effort)

Typical usage:
  python scripts/gpu/merge_lora.py \
    --adapter outputs/gpu/checkpoints/sft/final \
    --output outputs/gpu/merged_models/translation_sft_merged

If `--base-model` is omitted, the script tries to infer it from `adapter_config.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

console = Console()


@dataclass(frozen=True)
class MergeConfig:
    adapter_path: Path
    output_dir: Path
    base_model: str | None
    dtype: str
    device_map: str
    trust_remote_code: bool
    safe_serialization: bool
    max_shard_size: str
    offload_folder: Path | None


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def infer_base_model_from_adapter(adapter_path: Path) -> str | None:
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    try:
        adapter_config = _read_json(adapter_config_path)
    except Exception:
        return None

    base = adapter_config.get("base_model_name_or_path")
    return str(base) if base else None


def resolve_dtype(dtype: str):
    import torch

    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def merge_lora(cfg: MergeConfig) -> None:
    from peft import PeftModel
    from transformers import AutoTokenizer

    base_model = cfg.base_model or infer_base_model_from_adapter(cfg.adapter_path)
    if not base_model:
        raise ValueError(
            "Base model not provided and could not be inferred from adapter_config.json. "
            "Pass --base-model explicitly."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            "[bold green]Merging LoRA Adapter[/bold green]\n"
            f"Base model: {base_model}\n"
            f"Adapter: {cfg.adapter_path}\n"
            f"Output: {cfg.output_dir}",
            title="merge_lora",
        )
    )

    processor = None
    tokenizer = None
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code=cfg.trust_remote_code
        )
    except Exception:
        processor = None

    if processor is None:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=cfg.trust_remote_code
        )

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    torch_dtype = resolve_dtype(cfg.dtype)
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype

    if cfg.device_map:
        model_kwargs["device_map"] = cfg.device_map

    if cfg.offload_folder is not None:
        cfg.offload_folder.mkdir(parents=True, exist_ok=True)
        model_kwargs["offload_folder"] = str(cfg.offload_folder)

    model_family = "causal_lm"
    try:
        from transformers import AutoModelForVision2Seq as AutoModel

        model = AutoModel.from_pretrained(base_model, **model_kwargs)
        model_family = "vision2seq"
    except Exception:
        from transformers import AutoModelForCausalLM as AutoModel

        model = AutoModel.from_pretrained(base_model, **model_kwargs)

    model = PeftModel.from_pretrained(model, str(cfg.adapter_path))

    console.print("[blue]Merging weights (merge_and_unload)...[/blue]")
    model = model.merge_and_unload()

    console.print("[blue]Saving merged model...[/blue]")
    model.save_pretrained(
        str(cfg.output_dir),
        safe_serialization=cfg.safe_serialization,
        max_shard_size=cfg.max_shard_size,
    )

    if processor is not None:
        try:
            processor.save_pretrained(str(cfg.output_dir))
        except Exception:
            pass
    if tokenizer is not None:
        tokenizer.save_pretrained(str(cfg.output_dir))

    metadata = {
        "base_model": base_model,
        "adapter_path": str(cfg.adapter_path),
        "output_dir": str(cfg.output_dir),
        "model_family": model_family,
        "saved_processor": processor is not None,
        "saved_tokenizer": tokenizer is not None,
        "dtype": cfg.dtype,
        "device_map": cfg.device_map,
        "trust_remote_code": cfg.trust_remote_code,
        "safe_serialization": cfg.safe_serialization,
        "max_shard_size": cfg.max_shard_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(cfg.output_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    console.print("[bold green]Done.[/bold green]")


def parse_args(argv: list[str] | None = None) -> MergeConfig:
    parser = argparse.ArgumentParser(description="Merge PEFT LoRA adapter into base model")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to PEFT adapter directory (e.g., outputs/.../final)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged model (HF format)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model id/path; inferred from adapter_config.json if omitted",
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("LLM_FT_MERGE_DTYPE", "bfloat16"),
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype to load base model (default: bfloat16)",
    )
    parser.add_argument(
        "--device-map",
        default=os.environ.get("LLM_FT_MERGE_DEVICE_MAP", "auto"),
        help='Device map for loading model (default: "auto")',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Enable trust_remote_code when loading models (default: true)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code",
    )
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        default=True,
        help="Save safetensors (default: true)",
    )
    parser.add_argument(
        "--no-safe-serialization",
        dest="safe_serialization",
        action="store_false",
        help="Disable safetensors output",
    )
    parser.add_argument(
        "--max-shard-size",
        default=os.environ.get("LLM_FT_MAX_SHARD_SIZE", "5GB"),
        help='Max shard size for saving (default: "5GB")',
    )
    parser.add_argument(
        "--offload-folder",
        default=None,
        help="Optional offload folder for accelerate device_map offloading",
    )

    args = parser.parse_args(argv)
    adapter_path = Path(args.adapter)
    output_dir = Path(args.output)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    offload_folder = Path(args.offload_folder) if args.offload_folder else None

    return MergeConfig(
        adapter_path=adapter_path,
        output_dir=output_dir,
        base_model=args.base_model,
        dtype=args.dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
        offload_folder=offload_folder,
    )


def main() -> None:
    cfg = parse_args()
    merge_lora(cfg)


if __name__ == "__main__":
    main()
