#!/usr/bin/env python3
"""
Multimodal (Vision-Language) inference helper.

Reads a JSONL file containing either:
- `messages` (preferred) with multimodal user content:
    {"role":"user","content":[{"type":"image","image":"path"}, {"type":"text","text":"..."}]}
or:
- flat fields: `image`, `question`

Writes JSONL with an added `generated` field, similar to `inference_unified.py`.

Usage:
    python scripts/gpu/inference_vlm.py \
        --model Qwen/Qwen2-VL-7B-Instruct \
        --input data/processed/medical_vqa/test.jsonl \
        --output outputs/predictions_vlm.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from rich.console import Console
from rich.progress import track

from scripts.gpu.inference_unified import load_inference_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def _maybe_get_chat_template_applier(processor):
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template
    return None


def _extract_image_and_question(sample: dict) -> tuple[str, str, list[dict] | None]:
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
            return image_path, question, messages

    # Flat fallback
    image_path = str(sample.get("image", "")).strip()
    question = str(sample.get("question", "")).strip()
    if not image_path or not question:
        raise ValueError(
            "Sample missing image/question (expected messages or image/question fields)"
        )
    return image_path, question, None


def load_model(
    model_name: str,
    adapter_path: str | None,
    dtype: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import AutoModelForVision2Seq as AutoModel
    except Exception:  # pragma: no cover
        from transformers import AutoModelForCausalLM as AutoModel  # type: ignore

    bnb_config = None
    if load_in_4bit or load_in_8bit:
        compute_dtype = torch.float16
        if dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["device_map"] = "auto"

    if dtype in {"float16", "bfloat16"}:
        model_kwargs["torch_dtype"] = getattr(torch, dtype)

    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        console.print(f"[blue]Loaded adapter: {adapter_path}[/blue]")

    model.eval()
    return model, processor


def generate_one(
    model,
    processor,
    sample: dict,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    image_path, question, messages = _extract_image_and_question(sample)
    img = Image.open(image_path).convert("RGB")

    apply_chat_template = _maybe_get_chat_template_applier(processor)
    if messages and apply_chat_template is not None:
        prompt_messages = messages[:-1]  # drop assistant answer if present
        prompt = apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    elif apply_chat_template is not None:
        prompt = apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = question

    inputs = processor(text=prompt, images=img, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": int(max_tokens),
        "do_sample": bool(do_sample),
        "temperature": float(temperature) if do_sample else None,
        "top_p": float(top_p) if do_sample else None,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = int(inputs["input_ids"].shape[1])
    new_tokens = output_ids[0][prompt_len:]
    decoded = processor.decode(new_tokens, skip_special_tokens=True)
    return str(decoded).strip()


def main() -> None:
    config_path = None
    try:
        from scripts.gpu.inference_unified import (
            _extract_inference_config_path as _extract,  # type: ignore
        )

        config_path = _extract(sys.argv)
    except Exception:
        config_path = None

    defaults = {}
    if config_path:
        defaults = load_inference_config(config_path)

    parser = argparse.ArgumentParser(description="VLM inference (image + text -> text)")
    parser.add_argument(
        "--inference-config",
        type=str,
        default=config_path,
        help="YAML config (configs/inference/*.yaml)",
    )
    parser.add_argument(
        "--model", type=str, default=defaults.get("model"), help="HF model ID or local path"
    )
    parser.add_argument(
        "--adapter", type=str, default=defaults.get("adapter"), help="LoRA adapter path"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=defaults.get("input"), help="Input JSONL"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=defaults.get("output", "outputs/predictions_vlm.jsonl")
    )
    parser.add_argument("--max-tokens", type=int, default=int(defaults.get("max_tokens", 64)))
    parser.add_argument(
        "--temperature", type=float, default=float(defaults.get("temperature", 0.2))
    )
    parser.add_argument("--top-p", type=float, default=float(defaults.get("top_p", 0.95)))
    parser.add_argument(
        "--dtype",
        type=str,
        default=str(defaults.get("dtype", "auto")),
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--load-4bit", action="store_true", default=bool(defaults.get("load_4bit", False))
    )
    parser.add_argument(
        "--load-8bit", action="store_true", default=bool(defaults.get("load_8bit", False))
    )
    args = parser.parse_args()

    if not args.model:
        console.print("[red]Missing --model (or set model in --inference-config)[/red]")
        sys.exit(2)
    if not args.input:
        console.print("[red]Missing --input[/red]")
        sys.exit(2)

    model, processor = load_model(
        args.model,
        adapter_path=args.adapter,
        dtype=args.dtype,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
    )

    inputs: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fout:
        for item in track(inputs, description="Generating (VLM)"):
            try:
                generated = generate_one(
                    model=model,
                    processor=processor,
                    sample=item,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as e:
                logger.warning(f"Failed sample id={item.get('id')}: {e}")
                generated = ""

            fout.write(json.dumps({**item, "generated": generated}, ensure_ascii=False) + "\n")

    console.print(f"[green]Saved: {out_path}[/green]")


if __name__ == "__main__":
    main()
