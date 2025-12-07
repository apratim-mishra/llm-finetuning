#!/usr/bin/env python3
"""
GPU GRPO Training Script
Group Relative Policy Optimization for math reasoning (Use Case 3).
Online RL method - generates multiple responses per prompt and learns from relative rankings.

Supports:
- QLoRA (4-bit quantization with LoRA)
- DeepSpeed ZeRO Stage 2/3
- FSDP (Fully Sharded Data Parallel)
- Wandb and MLflow logging
- Custom reward functions from src/rewards/

Usage:
    python scripts/gpu/train_grpo.py --config configs/gpu/grpo_config.yaml

    # With DeepSpeed
    accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json \
        scripts/gpu/train_grpo.py --config configs/gpu/grpo_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import torch
import yaml
from rich.console import Console
from rich.panel import Panel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
console = Console()


def load_config(config_path: str) -> dict:
    """Load YAML configuration with environment variable expansion."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    def expand_env(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        elif isinstance(obj, dict):
            return {k: expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env(item) for item in obj]
        return obj

    return expand_env(config)


def setup_logging(config: dict, run_name: Optional[str] = None) -> dict:
    """Initialize Wandb and/or MLflow logging."""
    logging_status = {"wandb": False, "mlflow": False}

    # Wandb setup
    wandb_config = config.get("wandb", {})
    if wandb_config.get("enabled", False) or wandb_config:
        try:
            import wandb

            if not run_name:
                run_name = wandb_config.get(
                    "run_name",
                    f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )

            wandb.init(
                project=wandb_config.get("project", "llm-finetuning"),
                name=run_name,
                tags=wandb_config.get("tags", ["grpo", "rl"]),
                config=config,
                resume="allow",
            )
            logging_status["wandb"] = True
            console.print("[green]Wandb initialized[/green]")
        except ImportError:
            logger.warning("wandb not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")

    # MLflow setup
    mlflow_config = config.get("mlflow", {})
    if mlflow_config.get("enabled", False):
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "mlruns"))
            mlflow.set_experiment(mlflow_config.get("experiment_name", "grpo-training"))

            mlflow.start_run(run_name=run_name or f"grpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            mlflow.log_params({
                "model": config.get("model", {}).get("name"),
                "learning_rate": config.get("training", {}).get("learning_rate"),
                "kl_coef": config.get("grpo", {}).get("kl_coef"),
            })
            logging_status["mlflow"] = True
            console.print("[green]MLflow initialized[/green]")
        except ImportError:
            logger.warning("mlflow not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize mlflow: {e}")

    return logging_status


def get_model_and_tokenizer(config: dict):
    """Load model with quantization and proper attention implementation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_config = config.get("model", {})
    quant_config = config.get("quantization", {})

    model_name = model_config.get("name")
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

    # Check flash attention availability
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

    # Only set device_map if not using DeepSpeed/FSDP
    if not os.environ.get("ACCELERATE_USE_DEEPSPEED") and not os.environ.get("ACCELERATE_USE_FSDP"):
        model_kwargs["device_map"] = model_config.get("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # GRPO uses left padding for generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    console.print(f"[green]Model loaded: {model.num_parameters() / 1e9:.2f}B parameters[/green]")
    return model, tokenizer


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


def get_reward_function(config: dict) -> Callable:
    """Get reward function from config or src/rewards/."""
    grpo_config = config.get("grpo", {})
    reward_type = grpo_config.get("reward_function", "accuracy")

    try:
        # Try to import from src/rewards/
        from src.rewards.math_reward import get_reward_function as get_math_reward
        reward_fn = get_math_reward(reward_type)
        console.print(f"[green]Using reward function: {reward_type} from src/rewards/[/green]")
        return reward_fn
    except ImportError:
        logger.warning("Could not import from src/rewards/, using built-in reward")

    # Fallback built-in reward functions
    def accuracy_reward(completions: List[str], ground_truths: List[str], **kwargs) -> List[float]:
        """Simple accuracy-based reward."""
        import re

        rewards = []
        for completion, gt in zip(completions, ground_truths):
            # Extract answer patterns
            patterns = [
                r"[Aa]nswer:?\s*\$?([+-]?[\d,]+\.?\d*)",
                r"####\s*\$?([+-]?[\d,]+\.?\d*)",
                r"\\boxed\{([^}]+)\}",
            ]

            predicted = ""
            for pattern in patterns:
                match = re.search(pattern, completion)
                if match:
                    predicted = match.group(1).replace(",", "").strip()
                    break

            if not predicted:
                numbers = re.findall(r"[+-]?\d+\.?\d*", completion)
                predicted = numbers[-1] if numbers else ""

            try:
                pred_val = float(predicted) if predicted else None
                true_val = float(str(gt).replace(",", "").replace("$", ""))

                if pred_val is not None and abs(pred_val - true_val) < 0.01:
                    rewards.append(1.0)
                else:
                    # Partial credit for showing work
                    has_steps = any(x in completion.lower() for x in ["step", "first", "then", "="])
                    rewards.append(0.3 if has_steps else 0.0)
            except (ValueError, TypeError):
                rewards.append(0.0)

        return rewards

    return accuracy_reward


def load_grpo_dataset(config: dict):
    """Load prompts for GRPO training with ground truth answers."""
    from datasets import Dataset

    data_config = config.get("data", {})
    train_file = data_config.get("train_file")

    if not train_file or not Path(train_file).exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")

    data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Support multiple formats
                prompt = item.get("prompt") or item.get("question", "")
                ground_truth = item.get("ground_truth_answer") or item.get("answer", "")

                if prompt:
                    data.append({
                        "prompt": prompt,
                        "ground_truth_answer": ground_truth,
                    })

    console.print(f"[green]Loaded {len(data)} GRPO training prompts[/green]")
    return Dataset.from_list(data)


def train(config: dict):
    """Main GRPO training function."""
    from trl import GRPOTrainer, GRPOConfig

    console.print(Panel.fit(
        "[bold green]GPU GRPO Training[/bold green]\n"
        f"Model: {config.get('model', {}).get('name', 'N/A')}\n"
        f"KL Coef: {config.get('grpo', {}).get('kl_coef', 0.05)}\n"
        f"Generations: {config.get('grpo', {}).get('num_generations', 4)}",
        title="Starting Training"
    ))

    # Setup logging
    logging_status = setup_logging(config)

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(config)

    # PEFT config
    peft_config = get_peft_config(config)

    # Get reward function
    reward_fn = get_reward_function(config)

    # GRPO specific settings
    grpo_config = config.get("grpo", {})
    train_config = config.get("training", {})

    # Gradient checkpointing kwargs
    gc_kwargs = train_config.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    # Build report_to list
    report_to = []
    if logging_status["wandb"]:
        report_to.append("wandb")
    if logging_status["mlflow"]:
        report_to.append("mlflow")
    if not report_to:
        report_to = ["none"]

    # Training arguments
    training_args = GRPOConfig(
        output_dir=train_config.get("output_dir", "outputs/gpu/checkpoints/grpo"),
        # Training
        num_train_epochs=train_config.get("num_train_epochs", 1),
        max_steps=train_config.get("max_steps", -1),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
        # Optimizer - GRPO uses very low LR
        learning_rate=train_config.get("learning_rate", 1e-6),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_config.get("warmup_ratio", 0.1),
        weight_decay=train_config.get("weight_decay", 0.01),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        optim=train_config.get("optim", "adamw_torch"),
        # Precision
        bf16=train_config.get("bf16", True),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=gc_kwargs,
        # GRPO specific
        num_generations=grpo_config.get("num_generations", 4),
        max_completion_length=grpo_config.get("max_completion_length", 512),
        max_prompt_length=grpo_config.get("max_prompt_length", 512),
        temperature=grpo_config.get("temperature", 0.7),
        # Logging
        logging_steps=train_config.get("logging_steps", 5),
        logging_first_step=True,
        report_to=report_to,
        # Checkpointing
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 100),
        save_total_limit=train_config.get("save_total_limit", 3),
        # Misc
        seed=train_config.get("seed", 42),
        run_name=config.get("wandb", {}).get("run_name"),
        deepspeed=train_config.get("deepspeed"),
    )

    # Load dataset
    train_dataset = load_grpo_dataset(config)

    # Create reward function wrapper for TRL
    def reward_wrapper(completions: List[str], prompts: List[str] = None, **kwargs) -> List[float]:
        """Wrapper to adapt reward function to TRL interface."""
        ground_truths = kwargs.get("ground_truth_answer", [""] * len(completions))
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths] * len(completions)
        return reward_fn(completions, ground_truths)

    # Create GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_wrapper,
    )

    # Print training info
    console.print("\n[bold]GRPO Configuration:[/bold]")
    console.print(f"  Generations per prompt: {grpo_config.get('num_generations', 4)}")
    console.print(f"  Temperature: {grpo_config.get('temperature', 0.7)}")
    console.print(f"  Max completion length: {grpo_config.get('max_completion_length', 512)}")
    console.print(f"  Learning rate: {train_config.get('learning_rate', 1e-6)}")
    console.print(f"  Batch size: {train_config.get('per_device_train_batch_size', 2)}")
    console.print(f"  Effective batch: {train_config.get('per_device_train_batch_size', 2) * train_config.get('gradient_accumulation_steps', 8)}")

    console.print("\n[bold]Starting GRPO training...[/bold]")

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

    console.print(f"\n[bold green]GRPO training complete![/bold green]")
    console.print(f"  Model saved to: {final_output_dir}")
    console.print(f"  Final loss: {metrics.get('train_loss', 'N/A')}")

    # Log final metrics to MLflow
    if logging_status["mlflow"]:
        try:
            import mlflow
            mlflow.log_metrics(metrics)
            mlflow.end_run()
        except Exception:
            pass

    # Finish wandb
    if logging_status["wandb"]:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="GPU GRPO Training with TRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single GPU training
    python scripts/gpu/train_grpo.py --config configs/gpu/grpo_config.yaml

    # Multi-GPU with DeepSpeed
    accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json \\
        scripts/gpu/train_grpo.py --config configs/gpu/grpo_config.yaml
        """
    )
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed")
    args = parser.parse_args()

    if not Path(args.config).exists():
        console.print(f"[red]Config not found: {args.config}[/red]")
        sys.exit(1)

    config = load_config(args.config)

    if args.resume:
        config["training"]["resume_from_checkpoint"] = args.resume

    train(config)


if __name__ == "__main__":
    main()
