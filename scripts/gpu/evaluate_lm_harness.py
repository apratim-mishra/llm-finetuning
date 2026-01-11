#!/usr/bin/env python3
"""
LLM Evaluation Harness Integration
Comprehensive evaluation using EleutherAI's lm-evaluation-harness.

Supports:
- Standard benchmarks (GSM8K, MATH, ARC, HellaSwag, MMLU, etc.)
- Custom evaluation tasks
- HuggingFace models with LoRA adapters
- vLLM backend for faster evaluation
- Result logging to Wandb/MLflow

Usage:
    # Evaluate on GSM8K
    python scripts/gpu/evaluate_lm_harness.py --model outputs/gpu/checkpoints/sft/final \
        --tasks gsm8k --output outputs/eval_results.json

    # Multiple tasks
    python scripts/gpu/evaluate_lm_harness.py --model Qwen/Qwen2.5-7B-Instruct \
        --tasks gsm8k,arc_easy,hellaswag --num-fewshot 5

    # With vLLM backend (faster)
    python scripts/gpu/evaluate_lm_harness.py --model Qwen/Qwen2.5-7B-Instruct \
        --tasks gsm8k --backend vllm
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


# Task categories for organized evaluation
TASK_CATEGORIES = {
    "math": [
        "gsm8k",
        "minerva_math",
        "math_algebra",
        "math_geometry",
        "mathqa",           # Multiple choice math
        "asdiv",            # Diverse math word problems
    ],
    "math_advanced": [
        "hendrycks_math",   # MATH dataset (competition level)
        "aime_2024",        # American Invitational Math Exam
        "amc_2023",         # AMC competition problems
    ],
    "reasoning": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "piqa",
        "boolq",            # Boolean QA
        "commonsense_qa",   # Commonsense reasoning
        "logiqa",           # Logical reasoning
        "siqa",             # Social IQA
    ],
    "knowledge": ["mmlu", "triviaqa", "naturalqs"],
    "code": ["humaneval", "mbpp"],
    "translation": [],  # Custom tasks via --custom-eval
}

# Recommended few-shot settings per task
RECOMMENDED_FEWSHOT = {
    # Math
    "gsm8k": 8,
    "mathqa": 5,
    "asdiv": 5,
    "hendrycks_math": 4,
    "aime_2024": 0,
    "amc_2023": 0,
    # Reasoning
    "arc_easy": 25,
    "arc_challenge": 25,
    "hellaswag": 10,
    "winogrande": 5,
    "piqa": 5,
    "boolq": 0,
    "commonsense_qa": 7,
    "logiqa": 0,
    "siqa": 0,
    # Knowledge
    "mmlu": 5,
    # Code
    "humaneval": 0,
    "mbpp": 3,
}


def check_lm_eval_installed():
    """Check if lm-evaluation-harness is installed."""
    try:
        import lm_eval
        return True
    except ImportError:
        console.print("[red]lm-evaluation-harness not installed.[/red]")
        console.print("Install with: pip install lm-eval")
        console.print("Or: pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git")
        return False


def list_available_tasks():
    """List all available evaluation tasks."""
    try:
        from lm_eval import tasks
        available = tasks.TaskManager().all_tasks
        return sorted(available)
    except Exception as e:
        logger.error(f"Could not list tasks: {e}")
        return []


def run_evaluation(
    model_path: str,
    tasks: List[str],
    num_fewshot: Optional[int] = None,
    batch_size: int = 8,
    device: str = "cuda",
    backend: str = "hf",
    adapter_path: Optional[str] = None,
    limit: Optional[int] = None,
    output_path: Optional[str] = None,
    log_samples: bool = False,
) -> Dict:
    """Run lm-evaluation-harness evaluation."""
    from lm_eval import evaluator

    console.print(f"[bold green]Running LM Evaluation[/bold green]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Tasks: {', '.join(tasks)}")
    console.print(f"  Backend: {backend}")

    # Build model arguments (simple_evaluate creates the model internally)
    if backend == "vllm":
        model_args = f"pretrained={model_path},tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9"
        if adapter_path and Path(adapter_path).exists():
            model_args += f",enable_lora=True,max_lora_rank=64,lora_local_path={adapter_path}"
    else:
        model_args = f"pretrained={model_path},dtype=auto,trust_remote_code=True"
        if adapter_path and Path(adapter_path).exists():
            model_args += f",peft={adapter_path}"

    # Display recommended few-shot settings per task (informational only)
    # Note: simple_evaluate uses a single num_fewshot for all tasks.
    # For per-task few-shot, use lm_eval.evaluator.evaluate() with TaskConfig.
    console.print("\n[bold]Recommended few-shot settings (reference):[/bold]")
    for task in tasks:
        shots = num_fewshot if num_fewshot is not None else RECOMMENDED_FEWSHOT.get(task, 0)
        console.print(f"  {task}: {shots}-shot" + (" (using --num-fewshot override)" if num_fewshot is not None else " (recommended)"))

    # Run evaluation
    console.print("\n[bold]Starting evaluation...[/bold]")

    results = evaluator.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        log_samples=log_samples,
    )

    return results


def display_results(results: Dict):
    """Display evaluation results in a table."""
    if "results" not in results:
        console.print("[red]No results found[/red]")
        return

    table = Table(title="Evaluation Results")
    table.add_column("Task", style="cyan")
    table.add_column("Metric", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Stderr", style="dim")

    for task_name, task_results in results["results"].items():
        for metric_name, value in task_results.items():
            if metric_name.endswith(",none"):
                # Skip duplicate entries
                continue
            if isinstance(value, (int, float)):
                stderr = task_results.get(f"{metric_name}_stderr", "")
                stderr_str = f"+/- {stderr:.4f}" if isinstance(stderr, float) else ""
                table.add_row(
                    task_name,
                    metric_name,
                    f"{value:.4f}" if isinstance(value, float) else str(value),
                    stderr_str
                )

    console.print(table)


def save_results(results: Dict, output_path: str):
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "lm_eval_version": get_lm_eval_version(),
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]Results saved to: {output_path}[/green]")


def get_lm_eval_version() -> str:
    """Get lm-evaluation-harness version."""
    try:
        import lm_eval
        return getattr(lm_eval, "__version__", "unknown")
    except Exception:
        return "unknown"


def log_to_wandb(results: Dict, project: str = "llm-finetuning", run_name: Optional[str] = None):
    """Log results to Weights & Biases."""
    try:
        import wandb

        if not run_name:
            run_name = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        wandb.init(project=project, name=run_name, tags=["evaluation"])

        # Log metrics
        if "results" in results:
            metrics = {}
            for task_name, task_results in results["results"].items():
                for metric_name, value in task_results.items():
                    if isinstance(value, (int, float)) and not metric_name.endswith("_stderr"):
                        metrics[f"{task_name}/{metric_name}"] = value

            wandb.log(metrics)

        # Log config
        if "config" in results:
            wandb.config.update(results["config"])

        wandb.finish()
        console.print("[green]Results logged to Wandb[/green]")
    except ImportError:
        logger.warning("wandb not installed, skipping logging")
    except Exception as e:
        logger.warning(f"Failed to log to wandb: {e}")


def log_to_mlflow(results: Dict, experiment_name: str = "llm-evaluation"):
    """Log results to MLflow."""
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log metrics
            if "results" in results:
                for task_name, task_results in results["results"].items():
                    for metric_name, value in task_results.items():
                        if isinstance(value, (int, float)) and not metric_name.endswith("_stderr"):
                            mlflow.log_metric(f"{task_name}_{metric_name}", value)

            # Log config as params
            if "config" in results:
                for key, value in results["config"].items():
                    if isinstance(value, (str, int, float, bool)):
                        mlflow.log_param(key, value)

        console.print("[green]Results logged to MLflow[/green]")
    except ImportError:
        logger.warning("mlflow not installed, skipping logging")
    except Exception as e:
        logger.warning(f"Failed to log to mlflow: {e}")


def run_custom_translation_eval(
    model_path: str,
    test_file: str,
    adapter_path: Optional[str] = None,
    batch_size: int = 8,
) -> Dict:
    """Run custom translation evaluation using our metrics."""
    console.print("[bold]Running Translation Evaluation[/bold]")

    try:
        from src.evaluation.translation_metrics import evaluate_translation
    except ImportError:
        console.print("[red]Could not import translation metrics from src/[/red]")
        return {}

    # Load test data
    predictions = []
    references = []
    sources = []

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if "generated" in item:
                    predictions.append(item["generated"])
                if "reference" in item or "messages" in item:
                    # Extract reference from messages or direct field
                    if "messages" in item:
                        for msg in item["messages"]:
                            if msg["role"] == "assistant":
                                references.append(msg["content"])
                                break
                    else:
                        references.append(item.get("reference", ""))
                if "source" in item or "messages" in item:
                    if "messages" in item:
                        for msg in item["messages"]:
                            if msg["role"] == "user":
                                sources.append(msg["content"])
                                break
                    else:
                        sources.append(item.get("source", ""))

    if not predictions or not references:
        console.print("[red]No predictions/references found in test file[/red]")
        return {}

    # Run evaluation
    results = evaluate_translation(
        predictions=predictions,
        references=references,
        sources=sources if sources else None,
        metrics=["bleu", "chrf", "ter"],
    )

    return {"translation": results}


def run_custom_math_eval(
    model_path: str,
    test_file: str,
    adapter_path: Optional[str] = None,
) -> Dict:
    """Run custom math evaluation using our reward functions."""
    console.print("[bold]Running Math Evaluation[/bold]")

    try:
        from src.rewards.math_reward import evaluate_math_batch
    except ImportError:
        console.print("[red]Could not import math reward from src/[/red]")
        return {}

    # Load test data
    completions = []
    ground_truths = []

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if "generated" in item:
                    completions.append(item["generated"])
                if "answer" in item:
                    ground_truths.append(item["answer"])
                elif "ground_truth_answer" in item:
                    ground_truths.append(item["ground_truth_answer"])

    if not completions or not ground_truths:
        console.print("[red]No completions/ground_truths found in test file[/red]")
        return {}

    # Run evaluation
    results = evaluate_math_batch(completions, ground_truths)

    # Remove per-sample results for cleaner output
    if "results" in results:
        del results["results"]

    return {"math": results}


def main():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Harness Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available tasks
    python scripts/gpu/evaluate_lm_harness.py --list-tasks

    # Evaluate on GSM8K
    python scripts/gpu/evaluate_lm_harness.py --model outputs/gpu/checkpoints/sft/final \\
        --tasks gsm8k --output outputs/eval/gsm8k_results.json

    # Evaluate on multiple tasks
    python scripts/gpu/evaluate_lm_harness.py --model Qwen/Qwen2.5-7B-Instruct \\
        --tasks gsm8k,arc_easy,hellaswag --num-fewshot 5

    # With vLLM backend (faster)
    python scripts/gpu/evaluate_lm_harness.py --model Qwen/Qwen2.5-7B-Instruct \\
        --tasks gsm8k --backend vllm

    # Custom translation evaluation
    python scripts/gpu/evaluate_lm_harness.py --model outputs/gpu/checkpoints/sft/final \\
        --custom-eval translation --test-file outputs/predictions.jsonl
        """
    )

    parser.add_argument("--model", "-m", type=str, help="Model path or HuggingFace ID")
    parser.add_argument("--adapter", "-a", type=str, help="LoRA adapter path")
    parser.add_argument("--tasks", "-t", type=str, help="Comma-separated list of tasks")
    parser.add_argument("--num-fewshot", type=int, help="Number of few-shot examples")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"], help="Model backend")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--limit", type=int, help="Limit number of samples per task")
    parser.add_argument("--output", "-o", type=str, default="outputs/eval/results.json", help="Output JSON file")
    parser.add_argument("--log-samples", action="store_true", help="Log individual samples")
    parser.add_argument("--wandb", action="store_true", help="Log results to Wandb")
    parser.add_argument("--mlflow", action="store_true", help="Log results to MLflow")
    parser.add_argument("--wandb-project", type=str, default="llm-finetuning", help="Wandb project name")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--custom-eval", type=str, choices=["translation", "math"], help="Run custom evaluation")
    parser.add_argument("--test-file", type=str, help="Test file for custom evaluation")

    args = parser.parse_args()

    # List tasks mode
    if args.list_tasks:
        if not check_lm_eval_installed():
            sys.exit(1)

        tasks = list_available_tasks()
        console.print(f"[bold]Available Tasks ({len(tasks)}):[/bold]")

        # Group by category
        for category, category_tasks in TASK_CATEGORIES.items():
            console.print(f"\n[cyan]{category.upper()}:[/cyan]")
            for task in category_tasks:
                if task in tasks:
                    fewshot = RECOMMENDED_FEWSHOT.get(task, "?")
                    console.print(f"  - {task} (recommended: {fewshot}-shot)")

        console.print(f"\n[dim]Total available: {len(tasks)} tasks[/dim]")
        return

    # Custom evaluation mode
    if args.custom_eval:
        if not args.test_file:
            console.print("[red]--test-file required for custom evaluation[/red]")
            sys.exit(1)

        if args.custom_eval == "translation":
            results = run_custom_translation_eval(
                args.model or "",
                args.test_file,
                adapter_path=args.adapter,
            )
        else:
            results = run_custom_math_eval(
                args.model or "",
                args.test_file,
                adapter_path=args.adapter,
            )

        display_results({"results": results})
        if args.output:
            save_results({"results": results}, args.output)
        return

    # Standard lm-eval mode
    if not args.model:
        console.print("[red]--model required[/red]")
        sys.exit(1)

    if not args.tasks:
        console.print("[red]--tasks required (e.g., gsm8k,arc_easy)[/red]")
        sys.exit(1)

    if not check_lm_eval_installed():
        sys.exit(1)

    task_list = [t.strip() for t in args.tasks.split(",")]

    results = run_evaluation(
        model_path=args.model,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        backend=args.backend,
        adapter_path=args.adapter,
        limit=args.limit,
        log_samples=args.log_samples,
    )

    # Display and save results
    display_results(results)
    save_results(results, args.output)

    # Log to tracking systems
    if args.wandb:
        log_to_wandb(results, project=args.wandb_project)
    if args.mlflow:
        log_to_mlflow(results)


if __name__ == "__main__":
    main()
