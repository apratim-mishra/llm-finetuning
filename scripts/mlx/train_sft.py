#!/usr/bin/env python3
"""
MLX SFT Training Script
For Mac Apple Silicon (M1/M2/M3/M4) training using mlx-lm.
Supports Korean-English translation and Math reasoning use cases.

Features:
- LoRA fine-tuning with mlx-lm
- Wandb and MLflow logging
- Automatic data preparation
- Configurable via YAML

Usage:
    python scripts/mlx/train_sft.py --config configs/mlx/sft_korean_translation.yaml
    python scripts/mlx/train_sft.py --config configs/mlx/sft_math_reasoning.yaml
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore


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


def init_wandb(config: dict) -> bool:
    """Initialize Weights & Biases logging for MLX runs."""
    logging_config = config.get("logging", {})
    wandb_config = logging_config.get("wandb", {})

    if not wandb_config or not wandb_config.get("enabled", False):
        return False

    if wandb is None:
        console.print("[yellow]Warning: wandb not installed[/yellow]")
        return False

    run_name = wandb_config.get(
        "run_name",
        f"mlx-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )

    wandb.init(
        project=wandb_config.get("project", "mlx-training"),
        name=run_name,
        tags=wandb_config.get("tags", ["mlx", "sft"]),
        config=config,
        resume="allow",
    )
    console.print("[green]Wandb initialized[/green]")
    return True


def validate_data_format(filepath: str, max_samples: int = 5) -> bool:
    """Validate JSONL data format for MLX training."""
    try:
        from src.data.data_loader import validate_messages_format
        use_src_validation = True
    except ImportError:
        use_src_validation = False

    valid_count = 0
    invalid_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            try:
                item = json.loads(line)

                if use_src_validation:
                    if validate_messages_format(item):
                        valid_count += 1
                    else:
                        invalid_count += 1
                else:
                    # Basic validation
                    if "messages" in item or "text" in item:
                        valid_count += 1
                    else:
                        invalid_count += 1

                if i >= max_samples - 1:
                    break
            except json.JSONDecodeError:
                invalid_count += 1

    if invalid_count > 0:
        console.print(f"[yellow]Warning: {invalid_count} invalid samples found[/yellow]")

    return valid_count > 0


def prepare_mlx_data(data_path: str, output_dir: str, val_path: Optional[str] = None) -> tuple[str, str]:
    """
    Prepare data in MLX-LM expected format.
    MLX-LM expects train.jsonl and valid.jsonl in a directory.
    """
    data_dir = Path(output_dir) / "mlx_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_src = Path(data_path)
    train_dst = data_dir / "train.jsonl"

    # Validate source data
    if not train_src.exists():
        raise FileNotFoundError(f"Training file not found: {train_src}")

    if not validate_data_format(str(train_src)):
        console.print("[yellow]Warning: Data format validation failed[/yellow]")

    # Copy training data
    if not train_dst.exists() or train_dst.stat().st_mtime < train_src.stat().st_mtime:
        shutil.copy(train_src, train_dst)
        console.print(f"[green]Copied training data to {train_dst}[/green]")

    # Handle validation data
    val_dst = data_dir / "valid.jsonl"

    if val_path and Path(val_path).exists():
        val_src = Path(val_path)
    else:
        # Try to find validation file automatically
        val_src = train_src.parent / train_src.name.replace("train", "val")
        if not val_src.exists():
            val_src = train_src.parent / train_src.name.replace("train", "valid")

    if val_src.exists() and (not val_dst.exists() or val_dst.stat().st_mtime < val_src.stat().st_mtime):
        shutil.copy(val_src, val_dst)
        console.print(f"[green]Copied validation data to {val_dst}[/green]")

    return str(data_dir), str(train_dst)


def setup_logging(config: dict) -> dict:
    """Initialize Wandb and/or MLflow logging."""
    logging_status = {"wandb": False, "mlflow": False}

    # Wandb setup
    try:
        logging_status["wandb"] = init_wandb(config)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not init wandb: {e}[/yellow]")

    # MLflow setup
    logging_config = config.get("logging", {})
    mlflow_config = logging_config.get("mlflow", {})
    if mlflow_config.get("enabled", False):
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_config.get("tracking_uri", "mlruns"))
            mlflow.set_experiment(mlflow_config.get("experiment_name", "mlx-training"))

            mlflow.start_run(
                run_name=f"mlx-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            mlflow.log_params({
                "model": config.get("model", {}).get("name"),
                "lora_rank": config.get("lora", {}).get("rank"),
                "learning_rate": config.get("training", {}).get("learning_rate"),
            })
            logging_status["mlflow"] = True
            console.print("[green]MLflow initialized[/green]")
        except ImportError:
            console.print("[yellow]Warning: mlflow not installed[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not init mlflow: {e}[/yellow]")

    return logging_status


def build_mlx_command(config: dict, data_dir: str) -> list[str]:
    """Build mlx_lm.lora command from config."""
    model_config = config.get("model", {})
    lora_config = config.get("lora", {})
    training_config = config.get("training", {})
    checkpoint_config = config.get("checkpoint", {})

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_config.get("name", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
        "--data", data_dir,
        "--train",
        "--iters", str(training_config.get("iters", 1000)),
        "--batch-size", str(training_config.get("batch_size", 2)),
        "--lora-layers", str(lora_config.get("layers", 16)),
        "--lora-rank", str(lora_config.get("rank", 32)),
        "--learning-rate", str(training_config.get("learning_rate", 2e-5)),
        "--save-every", str(checkpoint_config.get("save_every", 200)),
        "--adapter-path", checkpoint_config.get("output_dir", "outputs/mlx/adapters"),
    ]

    # Optional: gradient checkpointing
    if training_config.get("grad_checkpoint", False):
        cmd.append("--grad-checkpoint")

    # Optional: validation batches
    val_config = config.get("validation", {})
    if val_config.get("val_batches"):
        cmd.extend(["--val-batches", str(val_config["val_batches"])])

    # Optional: max sequence length
    if model_config.get("max_seq_length"):
        cmd.extend(["--max-seq-length", str(model_config["max_seq_length"])])

    # Optional: seed
    if training_config.get("seed"):
        cmd.extend(["--seed", str(training_config["seed"])])

    return cmd


def parse_training_output(line: str, logging_status: dict):
    """Parse MLX training output and log metrics."""
    # MLX-LM output format: "Iter X: Train loss X.XXX, Val loss X.XXX, ..."
    if "Iter" not in line or "loss" not in line.lower():
        return

    try:
        parts = line.split()
        metrics = {}

        # Extract iteration
        for i, part in enumerate(parts):
            if part.startswith("Iter"):
                iter_str = parts[i + 1].rstrip(":")
                metrics["iteration"] = int(iter_str)
            elif "loss" in part.lower():
                # Next part should be the value
                if i + 1 < len(parts):
                    try:
                        loss_val = float(parts[i + 1].rstrip(","))
                        if "train" in part.lower() or "Train" in parts[i - 1]:
                            metrics["train/loss"] = loss_val
                        elif "val" in part.lower() or "Val" in parts[i - 1]:
                            metrics["val/loss"] = loss_val
                        else:
                            metrics["loss"] = loss_val
                    except ValueError:
                        pass

        # Log to wandb
        if logging_status.get("wandb") and metrics:
            try:
                import wandb
                wandb.log(metrics)
            except Exception:
                pass

        # Log to mlflow
        if logging_status.get("mlflow") and metrics:
            try:
                import mlflow
                step = metrics.pop("iteration", None)
                for key, value in metrics.items():
                    mlflow.log_metric(key.replace("/", "_"), value, step=step)
            except Exception:
                pass

    except Exception:
        pass


def run_training(config: dict):
    """Run MLX LoRA training."""
    console.print(Panel.fit(
        "[bold green]MLX LoRA Training[/bold green]\n"
        f"Model: {config.get('model', {}).get('name', 'N/A')}\n"
        f"Iterations: {config.get('training', {}).get('iters', 'N/A')}\n"
        f"LoRA Rank: {config.get('lora', {}).get('rank', 32)}",
        title="Starting Training"
    ))

    # Prepare data directory
    data_config = config.get("data", {})
    train_file = data_config.get("train_file", "data/processed/korean_english/train.jsonl")
    val_file = data_config.get("val_file")

    if not Path(train_file).exists():
        console.print(f"[red]Error: Training file not found: {train_file}[/red]")
        console.print("[yellow]Run data preparation first:[/yellow]")
        console.print("  python data/scripts/prepare_translation_data.py")
        console.print("  python data/scripts/prepare_math_data.py")
        return False

    data_dir, _ = prepare_mlx_data(
        train_file,
        config.get("checkpoint", {}).get("output_dir", "outputs/mlx/adapters"),
        val_path=val_file,
    )

    # Initialize logging
    logging_status = setup_logging(config)

    # Build command
    cmd = build_mlx_command(config, data_dir)

    console.print("\n[bold]Running command:[/bold]")
    console.print(" ".join(cmd))
    console.print()

    # Create output directory
    output_dir = Path(config.get("checkpoint", {}).get("output_dir", "outputs/mlx/adapters"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output
    config_save_path = output_dir / "training_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    # Run training
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output and log metrics
        for line in process.stdout:
            print(line, end='')
            parse_training_output(line, logging_status)

        process.wait()

        if process.returncode == 0:
            console.print("\n[bold green]Training completed successfully![/bold green]")
            console.print(f"  Adapters saved to: {output_dir}")

            # Log final info to mlflow
            if logging_status.get("mlflow"):
                try:
                    import mlflow
                    mlflow.log_artifact(str(config_save_path))
                    mlflow.end_run()
                except Exception:
                    pass

            return True
        else:
            console.print(f"\n[bold red]Training failed with code {process.returncode}[/bold red]")
            return False

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
        process.terminate()
        return False
    except Exception as e:
        console.print(f"\n[red]Error during training: {e}[/red]")
        return False
    finally:
        # Finish logging
        if logging_status.get("wandb"):
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

        if logging_status.get("mlflow"):
            try:
                import mlflow
                if mlflow.active_run():
                    mlflow.end_run()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="MLX SFT Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Korean-English translation
    python scripts/mlx/train_sft.py --config configs/mlx/sft_korean_translation.yaml

    # Math reasoning
    python scripts/mlx/train_sft.py --config configs/mlx/sft_math_reasoning.yaml

    # With overrides
    python scripts/mlx/train_sft.py --config configs/mlx/sft_korean_translation.yaml \\
        --model mlx-community/Qwen2.5-7B-Instruct-4bit --iters 500
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model name from config"
    )
    parser.add_argument(
        "--iters",
        type=int,
        help="Override number of iterations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate"
    )
    args = parser.parse_args()

    # Load config
    if not Path(args.config).exists():
        console.print(f"[red]Config file not found: {args.config}[/red]")
        sys.exit(1)

    config = load_config(args.config)

    # Apply overrides
    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.iters:
        config.setdefault("training", {})["iters"] = args.iters
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.learning_rate:
        config.setdefault("training", {})["learning_rate"] = args.learning_rate

    # Run training
    success = run_training(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
