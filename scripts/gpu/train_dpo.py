#!/usr/bin/env python3
"""
GPU DPO Training Script
Direct Preference Optimization for math reasoning (Use Case 3).
Requires SFT checkpoint as starting point.

Usage:
    python scripts/gpu/train_dpo.py --config configs/gpu/dpo_config.yaml

    # With DeepSpeed
    accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json \
        scripts/gpu/train_dpo.py --config configs/gpu/dpo_config.yaml
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

        if not run_name:
            run_name = wandb_config.get(
                "run_name",
                f"dpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

        wandb.init(
            project=wandb_config.get("project", "llm-finetuning"),
            name=run_name,
            tags=wandb_config.get("tags", ["dpo"]),
            config=config,
            resume="allow",
        )
        console.print("[green]Wandb initialized[/green]")
        return True
    except ImportError:
        logger.warning("wandb not installed")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return False


def get_model_and_tokenizer(config: dict):
    """Load base model (optionally with a LoRA adapter) and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config.get("model", {})
    quant_config = config.get("quantization", {})

    model_name = model_config.get("name")
    adapter_path = model_config.get("adapter") or model_config.get("adapter_path")
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
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )
        console.print("[green]4-bit quantization enabled (QLoRA)[/green]")

    # Check flash attention
    attn_impl = model_config.get("attn_implementation", "flash_attention_2")
    try:
        if attn_impl == "flash_attention_2":
            import flash_attn  # noqa
    except ImportError:
        logger.warning("flash-attn not installed, using eager attention")
        attn_impl = "eager"

    model_kwargs = {
        "quantization_config": bnb_config,
        "torch_dtype": getattr(torch, model_config.get("torch_dtype", "bfloat16")),
        "attn_implementation": attn_impl,
        "trust_remote_code": True,
    }

    if not os.environ.get("ACCELERATE_USE_DEEPSPEED") and not os.environ.get("ACCELERATE_USE_FSDP"):
        model_kwargs["device_map"] = model_config.get("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # DPO typically uses left padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    has_adapter = False
    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        has_adapter = True
        console.print(f"[blue]Loaded adapter (continuing): {adapter_path}[/blue]")

    console.print(f"[green]Model loaded: {model.num_parameters() / 1e9:.2f}B parameters[/green]")
    return model, tokenizer, has_adapter


def get_peft_config(config: dict):
    """Create PEFT LoRA configuration."""
    from peft import LoraConfig, TaskType

    lora_config = config.get("lora", {})

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
    )

    console.print(f"[green]LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}[/green]")
    return peft_config


def load_preference_dataset(config: dict):
    """Load preference pairs for DPO training."""
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
                    item = json.loads(line)
                    # Validate preference pair format
                    if all(k in item for k in ["prompt", "chosen", "rejected"]):
                        data.append(item)
        return Dataset.from_list(data)

    train_dataset = load_jsonl(train_file)
    val_dataset = None
    if val_file and Path(val_file).exists():
        val_dataset = load_jsonl(val_file)

    console.print(f"[green]Loaded {len(train_dataset)} preference pairs[/green]")
    if val_dataset:
        console.print(f"[green]Loaded {len(val_dataset)} validation pairs[/green]")

    return train_dataset, val_dataset


def train(config: dict):
    """Main DPO training function."""
    from trl import DPOTrainer, DPOConfig

    console.print(Panel.fit(
        "[bold green]GPU DPO Training[/bold green]\n"
        f"Model: {config.get('model', {}).get('name', 'N/A')}\n"
        f"Beta: {config.get('dpo', {}).get('beta', 0.1)}",
        title="Starting Training"
    ))

    # Setup wandb
    setup_wandb(config)

    # Load model and tokenizer (optionally continuing from an existing adapter)
    model, tokenizer, has_adapter = get_model_and_tokenizer(config)

    # PEFT config (create a new adapter only if one wasn't loaded)
    peft_config = None if has_adapter else get_peft_config(config)

    # DPO specific settings
    dpo_config = config.get("dpo", {})
    train_config = config.get("training", {})

    # Build training arguments
    gc_kwargs = train_config.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    training_args = DPOConfig(
        output_dir=train_config.get("output_dir", "outputs/gpu/checkpoints/dpo"),
        # DPO specific
        beta=dpo_config.get("beta", 0.1),
        loss_type=dpo_config.get("loss_type", "sigmoid"),
        label_smoothing=dpo_config.get("label_smoothing", 0.0),
        max_length=dpo_config.get("max_length", 2048),
        max_prompt_length=dpo_config.get("max_prompt_length", 1024),
        # Training
        num_train_epochs=train_config.get("num_train_epochs", 1),
        max_steps=train_config.get("max_steps", -1),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
        # Optimizer - DPO uses much lower LR
        learning_rate=train_config.get("learning_rate", 5e-7),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        weight_decay=train_config.get("weight_decay", 0.01),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        optim=train_config.get("optim", "adamw_torch"),
        # Precision
        bf16=train_config.get("bf16", True),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=gc_kwargs,
        # Evaluation
        eval_strategy=train_config.get("eval_strategy", "steps"),
        eval_steps=train_config.get("eval_steps", 100),
        # Logging
        logging_steps=train_config.get("logging_steps", 10),
        logging_first_step=True,
        report_to=train_config.get("report_to", ["wandb"]) if config.get("wandb") else [],
        # Checkpointing
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 100),
        save_total_limit=train_config.get("save_total_limit", 3),
        load_best_model_at_end=train_config.get("load_best_model_at_end", True),
        # Misc
        seed=train_config.get("seed", 42),
        run_name=config.get("wandb", {}).get("run_name"),
        deepspeed=train_config.get("deepspeed"),
    )

    # Load datasets
    train_dataset, val_dataset = load_preference_dataset(config)

    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPOTrainer creates reference model automatically
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Print training info
    console.print("\n[bold]DPO Configuration:[/bold]")
    console.print(f"  Beta (KL penalty): {dpo_config.get('beta', 0.1)}")
    console.print(f"  Loss type: {dpo_config.get('loss_type', 'sigmoid')}")
    console.print(f"  Learning rate: {train_config.get('learning_rate', 5e-7)}")
    console.print(f"  Epochs: {train_config.get('num_train_epochs', 1)}")
    console.print(f"  Batch size: {train_config.get('per_device_train_batch_size', 2)}")
    console.print(f"  Effective batch: {train_config.get('per_device_train_batch_size', 2) * train_config.get('gradient_accumulation_steps', 8)}")

    console.print("\n[bold]Starting DPO training...[/bold]")

    # Train
    train_result = trainer.train()

    # Save final model
    final_output_dir = Path(training_args.output_dir) / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    console.print(f"\n[bold green]DPO training complete![/bold green]")
    console.print(f"  Model saved to: {final_output_dir}")
    console.print(f"  Final loss: {metrics.get('train_loss', 'N/A'):.4f}")

    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="GPU DPO Training with TRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single GPU training
    python scripts/gpu/train_dpo.py --config configs/gpu/dpo_config.yaml

    # Multi-GPU with DeepSpeed
    accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json \\
        scripts/gpu/train_dpo.py --config configs/gpu/dpo_config.yaml
        """
    )
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Optional GPU profile YAML to merge on top of config (e.g., configs/gpu/profiles/a10_24gb.yaml)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed")
    args = parser.parse_args()

    if not Path(args.config).exists():
        console.print(f"[red]Config not found: {args.config}[/red]")
        sys.exit(1)

    from src.training.config_utils import load_config_with_profile

    config = load_config_with_profile(args.config, profile_path=args.profile)

    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume

    from src.training.sanity import require_cuda

    require_cuda(console=console)

    train(config)


if __name__ == "__main__":
    main()
