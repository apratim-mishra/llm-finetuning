#!/usr/bin/env python3
"""
GPU Vision-Language SFT Training Script (VLM)

Designed for multimodal VQA-style tasks (image + question -> answer), e.g. VQA-RAD.

This script intentionally does NOT use TRL SFTTrainer because most VLMs require a
processor-driven multimodal collator (images + text + labels masking).

Usage:
    python scripts/gpu/train_vlm_sft.py --config configs/gpu/sft_medical_vqa_vlm.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from rich.console import Console
from rich.panel import Panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def load_config(config_path: str) -> dict:
    from src.training.config_utils import load_config_with_profile

    return load_config_with_profile(config_path)


def setup_wandb(config: dict, run_name: str | None = None) -> bool:
    wandb_config = config.get("wandb", {})
    if not wandb_config:
        return False

    try:
        import wandb

        if not run_name:
            run_name = wandb_config.get(
                "run_name", f"vlm-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

        wandb.init(
            project=wandb_config.get("project", "llm-finetuning"),
            name=run_name,
            tags=wandb_config.get("tags", ["vlm", "sft"]),
            config=config,
            resume="allow",
        )
        console.print("[green]Wandb initialized[/green]")
        return True
    except ImportError:
        logger.warning("wandb not installed, skipping logging")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return False


def _maybe_get_chat_template_applier(processor):
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template
    return None


def get_model_and_processor(config: dict):
    from transformers import AutoProcessor, BitsAndBytesConfig

    model_config = config.get("model", {})
    quant_config = config.get("quantization", {})

    model_name = model_config.get("name")
    if not model_name:
        raise ValueError("Missing model.name in config")

    console.print(f"[blue]Loading VLM: {model_name}[/blue]")

    bnb_config = None
    if quant_config.get("enabled", False):
        compute_dtype = getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            load_in_8bit=quant_config.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )
        console.print("[green]Quantization enabled (bitsandbytes)[/green]")

    # Determine attention implementation
    attn_impl = model_config.get("attn_implementation", "flash_attention_2")
    try:
        if attn_impl == "flash_attention_2":
            import flash_attn  # noqa: F401
    except ImportError:
        logger.warning("flash-attn not installed, falling back to eager attention")
        attn_impl = "eager"

    # Some VLMs live under AutoModelForVision2Seq; keep a fallback for older transformers.
    try:
        from transformers import AutoModelForVision2Seq as AutoModel
    except Exception:  # pragma: no cover
        from transformers import AutoModelForCausalLM as AutoModel  # type: ignore

    model_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "torch_dtype": getattr(torch, model_config.get("torch_dtype", "bfloat16")),
        "attn_implementation": attn_impl,
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
    }

    # Only set device_map if not using DeepSpeed/FSDP under accelerate.
    if not os.environ.get("ACCELERATE_USE_DEEPSPEED") and not os.environ.get("ACCELERATE_USE_FSDP"):
        model_kwargs["device_map"] = model_config.get("device_map", "auto")

    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token exists when a tokenizer is available.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, processor


def get_peft_model(config: dict, model):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    lora_cfg = config.get("lora", {})

    target_modules = lora_cfg.get("target_modules", "all-linear")
    if isinstance(target_modules, str) and target_modules != "all-linear":
        target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 64)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        bias=str(lora_cfg.get("bias", "none")),
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=lora_cfg.get("modules_to_save"),
    )

    quant_enabled = bool(config.get("quantization", {}).get("enabled", False))
    if quant_enabled:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)
    console.print(f"[green]LoRA enabled: r={peft_config.r}, alpha={peft_config.lora_alpha}[/green]")
    return model


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_datasets(config: dict):
    from datasets import Dataset

    data_cfg = config.get("data", {})
    train_file = data_cfg.get("train_file")
    val_file = data_cfg.get("val_file")
    max_samples = data_cfg.get("max_samples")

    if not train_file or not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    train_rows = load_jsonl(train_file)
    if max_samples:
        train_rows = train_rows[: int(max_samples)]

    val_rows: list[dict] | None = None
    if val_file and Path(val_file).exists():
        val_rows = load_jsonl(val_file)

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows) if val_rows else None

    console.print(f"[green]Loaded train: {len(train_ds)}[/green]")
    if val_ds is not None:
        console.print(f"[green]Loaded val:   {len(val_ds)}[/green]")

    return train_ds, val_ds


@dataclass
class VLMDataCollator:
    processor: Any
    max_length: int | None = None

    def __post_init__(self) -> None:
        self._apply_chat_template = _maybe_get_chat_template_applier(self.processor)

    def _get_first_image_path(self, sample: dict) -> str:
        # Prefer multimodal messages if present.
        messages = sample.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if (
                            isinstance(part, dict)
                            and part.get("type") == "image"
                            and part.get("image")
                        ):
                            return str(part["image"])

        # Fallback to flat field
        if sample.get("image"):
            return str(sample["image"])

        raise ValueError(
            "Missing image path in sample (expected messages[].content image or 'image' field)"
        )

    def _build_messages(self, sample: dict) -> list[dict[str, Any]]:
        messages = sample.get("messages")
        if isinstance(messages, list) and messages:
            return messages

        # Fallback to minimal multimodal message format
        image_path = self._get_first_image_path(sample)
        question = str(sample.get("question", "")).strip()
        answer = str(sample.get("answer", "")).strip()
        system_prompt = str(sample.get("system_prompt") or "You are a helpful assistant.")
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": answer},
        ]

    def _apply_template(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        if self._apply_chat_template is None:
            # Very conservative fallback if chat templates are unavailable.
            user_text = ""
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if part.get("type") == "text":
                                user_text = part.get("text", "")
                    else:
                        user_text = str(content)
                    break
            assistant = "Answer:"
            return f"{user_text}\n{assistant}"

        return self._apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        texts_full: list[str] = []
        images: list[Image.Image] = []
        prompt_lens: list[int] = []

        for sample in features:
            messages_full = self._build_messages(sample)
            messages_prompt = messages_full[:-1]

            prompt_text = self._apply_template(messages_prompt, add_generation_prompt=True)
            full_text = self._apply_template(messages_full, add_generation_prompt=False)

            image_path = self._get_first_image_path(sample)
            img = Image.open(image_path).convert("RGB")
            images.append(img)
            texts_full.append(full_text)

            prompt_inputs = self.processor(
                text=prompt_text,
                images=img,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            prompt_lens.append(int(prompt_inputs["input_ids"].shape[1]))

        model_inputs = self.processor(
            text=texts_full,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()

        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            labels[attention_mask == 0] = -100

        for i, pl in enumerate(prompt_lens):
            labels[i, :pl] = -100

        model_inputs["labels"] = labels
        return model_inputs


def get_training_args(config: dict):
    from transformers import TrainingArguments

    train_cfg = config.get("training", {})
    output_dir = train_cfg.get("output_dir", "outputs/gpu/checkpoints/vlm_sft")

    report_to = train_cfg.get("report_to")
    if report_to is None:
        report_to = ["wandb"] if config.get("wandb") else []

    eval_strategy = train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "steps"))
    save_strategy = train_cfg.get("save_strategy", "steps")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        max_steps=int(train_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        warmup_steps=int(train_cfg.get("warmup_steps", 0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        optim=str(train_cfg.get("optim", "adamw_torch")),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        evaluation_strategy=str(eval_strategy),
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        logging_first_step=bool(train_cfg.get("logging_first_step", True)),
        report_to=report_to,
        save_strategy=str(save_strategy),
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        load_best_model_at_end=bool(train_cfg.get("load_best_model_at_end", True)),
        metric_for_best_model=str(train_cfg.get("metric_for_best_model", "eval_loss")),
        greater_is_better=bool(train_cfg.get("greater_is_better", False)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(train_cfg.get("dataloader_pin_memory", True)),
        seed=int(train_cfg.get("seed", 42)),
        run_name=config.get("wandb", {}).get("run_name"),
        deepspeed=train_cfg.get("deepspeed"),
        remove_unused_columns=False,
    )


def train(config: dict) -> None:
    from transformers import Trainer

    console.print(
        Panel.fit(
            "[bold green]GPU VLM SFT Training[/bold green]\n"
            f"Model: {config.get('model', {}).get('name', 'N/A')}\n"
            f"LoRA r: {config.get('lora', {}).get('r', 32)}",
            title="Starting Training",
        )
    )

    setup_wandb(config)

    model, processor = get_model_and_processor(config)
    model = get_peft_model(config, model)

    training_args = get_training_args(config)
    train_ds, val_ds = load_datasets(config)

    max_length = config.get("data", {}).get("max_seq_length")
    collator = VLMDataCollator(
        processor=processor, max_length=int(max_length) if max_length else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    console.print("\n[bold]Starting training...[/bold]")
    train_result = trainer.train()

    final_dir = Path(training_args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    try:
        processor.save_pretrained(str(final_dir))
    except Exception:
        pass

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    if val_ds is not None:
        metrics["eval_samples"] = len(val_ds)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    console.print(f"\n[bold green]Training complete![/bold green] Saved to: {final_dir}")

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU VLM SFT Training (Transformers + PEFT)")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to training config YAML"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional GPU profile YAML to merge on top of config",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        console.print(f"[red]Config not found: {args.config}[/red]")
        sys.exit(1)

    from src.training.config_utils import load_config_with_profile

    cfg = load_config_with_profile(args.config, profile_path=args.profile)

    from src.training.sanity import require_cuda

    require_cuda(console=console)

    train(cfg)


if __name__ == "__main__":
    main()
