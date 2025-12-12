#!/usr/bin/env python3
"""
GPU SFT Training Script
Uses TRL SFTTrainer with QLoRA for efficient fine-tuning.
Supports DeepSpeed and FSDP for multi-GPU training.

Usage:
    python scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml

    # With DeepSpeed
    accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json \
        scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from rich.panel import Panel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()


def load_config(config_path: str) -> dict:
    """Load YAML configuration with optional profile overrides."""
    from src.training.config_utils import load_config_with_profile

    return load_config_with_profile(config_path)


def setup_wandb(config: dict, run_name: Optional[str] = None) -> bool:
    """Initialize Weights & Biases logging."""
    wandb_config = config.get("wandb", {})
    if not wandb_config:
        return False

    try:
        import wandb

        # Generate run name if not provided
        if not run_name:
            run_name = wandb_config.get(
                "run_name",
                f"sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

        wandb.init(
            project=wandb_config.get("project", "llm-finetuning"),
            name=run_name,
            tags=wandb_config.get("tags", ["sft"]),
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


def get_model_and_tokenizer(config: dict):
    """Load model with quantization and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config.get("model", {})
    quant_config = config.get("quantization", {})

    model_name = model_config.get("name", "Qwen/Qwen2.5-7B-Instruct")
    console.print(f"[blue]Loading model: {model_name}[/blue]")

    # Build quantization config
    bnb_config = None
    if quant_config.get("enabled", False):
        compute_dtype = getattr(
            torch,
            quant_config.get("bnb_4bit_compute_dtype", "bfloat16")
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            load_in_8bit=quant_config.get("load_in_8bit", False),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )
        console.print("[green]4-bit quantization enabled (QLoRA)[/green]")

    # Determine attention implementation
    attn_impl = model_config.get("attn_implementation", "flash_attention_2")
    try:
        # Check if flash attention is available
        if attn_impl == "flash_attention_2":
            import flash_attn  # noqa
    except ImportError:
        logger.warning("flash-attn not installed, falling back to eager attention")
        attn_impl = "eager"

    # Load model
    model_kwargs = {
        "quantization_config": bnb_config,
        "torch_dtype": getattr(torch, model_config.get("torch_dtype", "bfloat16")),
        "attn_implementation": attn_impl,
        "trust_remote_code": model_config.get("trust_remote_code", True),
    }

    # Only set device_map if not using DeepSpeed/FSDP
    if not os.environ.get("ACCELERATE_USE_DEEPSPEED") and not os.environ.get("ACCELERATE_USE_FSDP"):
        model_kwargs["device_map"] = model_config.get("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",  # Required for SFT
    )

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    console.print(f"[green]Model loaded: {model.num_parameters() / 1e9:.2f}B parameters[/green]")

    return model, tokenizer


def get_peft_config(config: dict):
    """Create PEFT LoRA configuration."""
    from peft import LoraConfig, TaskType

    lora_config = config.get("lora", {})

    # Parse target modules
    target_modules = lora_config.get("target_modules", "all-linear")
    if isinstance(target_modules, str) and target_modules != "all-linear":
        target_modules = [m.strip() for m in target_modules.split(",")]

    peft_config = LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("lora_alpha", 128),
        lora_dropout=lora_config.get("lora_dropout", 0.05),
        target_modules=target_modules,
        bias=lora_config.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=lora_config.get("modules_to_save"),
    )

    console.print(f"[green]LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}[/green]")
    return peft_config


def get_training_args(config: dict):
    """Create training arguments compatible with SFTConfig."""
    from trl import SFTConfig

    train_config = config.get("training", {})
    sft_config = config.get("sft", {})

    # Handle gradient checkpointing kwargs
    gc_kwargs = train_config.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    # Build output directory with timestamp
    output_dir = train_config.get("output_dir", "outputs/gpu/checkpoints/sft")

    args = SFTConfig(
        output_dir=output_dir,
        # Training
        num_train_epochs=train_config.get("num_train_epochs", 3),
        max_steps=train_config.get("max_steps", -1),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
        # Optimizer
        learning_rate=train_config.get("learning_rate", 2e-5),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        warmup_steps=train_config.get("warmup_steps", 0),
        weight_decay=train_config.get("weight_decay", 0.01),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        optim=train_config.get("optim", "adamw_torch"),
        # Precision & Memory
        bf16=train_config.get("bf16", True),
        fp16=train_config.get("fp16", False),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=gc_kwargs,
        # Evaluation
        eval_strategy=train_config.get("eval_strategy", "steps"),
        eval_steps=train_config.get("eval_steps", 500),
        # Logging
        logging_steps=train_config.get("logging_steps", 10),
        logging_first_step=train_config.get("logging_first_step", True),
        report_to=train_config.get("report_to", ["wandb"]) if config.get("wandb") else [],
        # Checkpointing
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 500),
        save_total_limit=train_config.get("save_total_limit", 3),
        load_best_model_at_end=train_config.get("load_best_model_at_end", True),
        metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_config.get("greater_is_better", False),
        # Data loading
        dataloader_num_workers=train_config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=train_config.get("dataloader_pin_memory", True),
        # Misc
        seed=train_config.get("seed", 42),
        run_name=config.get("wandb", {}).get("run_name"),
        # DeepSpeed
        deepspeed=train_config.get("deepspeed"),
        # SFT specific
        max_seq_length=sft_config.get("max_seq_length", 2048),
        packing=sft_config.get("packing", True),
        dataset_text_field="text",
    )

    return args


def load_dataset_from_config(config: dict):
    """Load training and validation datasets from JSONL files."""
    from datasets import Dataset

    data_config = config.get("data", {})
    train_file = data_config.get("train_file")
    val_file = data_config.get("val_file")

    if not train_file or not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    def load_jsonl(path: str) -> Dataset:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return Dataset.from_list(data)

    train_dataset = load_jsonl(train_file)
    val_dataset = None
    if val_file and Path(val_file).exists():
        val_dataset = load_jsonl(val_file)

    console.print(f"[green]Loaded {len(train_dataset)} training samples[/green]")
    if val_dataset:
        console.print(f"[green]Loaded {len(val_dataset)} validation samples[/green]")

    return train_dataset, val_dataset


def get_formatting_func(format_type: str = "chatml"):
    """Get formatting function for different message formats."""

    def chatml_format(sample):
        """Format messages to ChatML text (Qwen, Mistral style)."""
        messages = sample.get("messages", [])
        text_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return {"text": "\n".join(text_parts)}

    def llama_format(sample):
        """Format for Llama-style chat."""
        messages = sample.get("messages", [])
        text_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text_parts.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                text_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                text_parts.append(content)

        return {"text": "\n".join(text_parts)}

    formatters = {
        "chatml": chatml_format,
        "llama": llama_format,
    }

    return formatters.get(format_type, chatml_format)


def train(config: dict):
    """Main training function."""
    from trl import SFTTrainer

    console.print(Panel.fit(
        "[bold green]GPU SFT Training[/bold green]\n"
        f"Model: {config.get('model', {}).get('name', 'N/A')}\n"
        f"LoRA rank: {config.get('lora', {}).get('r', 64)}",
        title="Starting Training"
    ))

    # Setup wandb
    setup_wandb(config)

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config)

    # Get PEFT config
    peft_config = get_peft_config(config)

    # Get training arguments
    training_args = get_training_args(config)

    # Load datasets
    train_dataset, val_dataset = load_dataset_from_config(config)

    # Format dataset
    sft_config = config.get("sft", {})
    format_type = sft_config.get("format_type", "chatml")
    formatting_func = get_formatting_func(format_type)

    train_dataset = train_dataset.map(formatting_func, remove_columns=train_dataset.column_names)
    if val_dataset:
        val_dataset = val_dataset.map(formatting_func, remove_columns=val_dataset.column_names)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
    )

    # Print training info
    console.print("\n[bold]Training Configuration:[/bold]")
    console.print(f"  Epochs: {training_args.num_train_epochs}")
    console.print(f"  Batch size: {training_args.per_device_train_batch_size}")
    console.print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    console.print(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    console.print(f"  Learning rate: {training_args.learning_rate}")
    console.print(f"  Max seq length: {training_args.max_seq_length}")
    console.print(f"  Packing: {training_args.packing}")

    console.print("\n[bold]Starting training...[/bold]")

    # Train
    train_result = trainer.train()

    # Save final model
    final_output_dir = Path(training_args.output_dir) / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    if val_dataset:
        metrics["eval_samples"] = len(val_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Model saved to: {final_output_dir}")
    console.print(f"  Final loss: {metrics.get('train_loss', 'N/A'):.4f}")

    # Finish wandb
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="GPU SFT Training with TRL and QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single GPU training
    python scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml

    # Multi-GPU with DeepSpeed
    accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json \\
        scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml

    # Multi-GPU with FSDP
    accelerate launch --config_file configs/gpu/fsdp/accelerate_fsdp.yaml \\
        scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional GPU profile YAML to merge on top of config (e.g., configs/gpu/profiles/a10_24gb.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by accelerate)"
    )
    args = parser.parse_args()

    # Load config
    if not Path(args.config).exists():
        console.print(f"[red]Config not found: {args.config}[/red]")
        sys.exit(1)

    from src.training.config_utils import load_config_with_profile

    config = load_config_with_profile(args.config, profile_path=args.profile)

    # Handle resume
    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume

    from src.training.sanity import require_cuda

    require_cuda(console=console)

    # Run training
    train(config)


if __name__ == "__main__":
    main()
