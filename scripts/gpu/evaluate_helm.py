#!/usr/bin/env python3
"""
HELM (Holistic Evaluation of Language Models) Integration
Stanford CRFM's comprehensive evaluation framework.

HELM evaluates models across:
- Accuracy: Multiple choice, open-ended, extraction
- Calibration: Confidence vs correctness
- Robustness: Perturbations, adversarial examples
- Fairness: Demographic parity
- Bias: Stereotypes, toxicity
- Efficiency: Inference time, memory

Scenarios supported:
- NarrativeQA, NaturalQuestions, QuAC (QA)
- HellaSwag, OpenBookQA, PIQA (Commonsense)
- BoolQ, COPA, RTE (NLI)
- GSM8K, MATH (Math)
- HumanEval, MBPP (Code)
- MMLU (Knowledge)
- TruthfulQA (Truthfulness)
- BBQ (Bias)
- And many more...

Usage:
    # Run HELM evaluation
    python scripts/gpu/evaluate_helm.py --model outputs/gpu/checkpoints/sft/final \
        --scenarios mmlu,gsm8k --output outputs/eval/helm_results

    # List available scenarios
    python scripts/gpu/evaluate_helm.py --list-scenarios

    # Run with specific metrics
    python scripts/gpu/evaluate_helm.py --model Qwen/Qwen2.5-7B-Instruct \
        --scenarios truthfulqa --metrics accuracy,calibration
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


# HELM scenarios organized by category
HELM_SCENARIOS = {
    "knowledge": [
        "mmlu",
        "wikitext_103",
        "wikifact",
    ],
    "reasoning": [
        "hellaswag",
        "openbookqa",
        "piqa",
        "siqa",
        "commonsenseqa",
        "winogrande",
    ],
    "math": [
        "gsm",  # GSM8K in HELM
        "math",
        "legalbench",
    ],
    "qa": [
        "narrativeqa",
        "naturalquestions",
        "quac",
        "squad",
        "triviaqa",
        "boolq",
    ],
    "truthfulness": [
        "truthfulqa",
    ],
    "code": [
        "humaneval",
        "mbpp",
        "apps",
    ],
    "bias": [
        "bbq",
        "bold",
    ],
    "toxicity": [
        "real_toxicity_prompts",
    ],
    "summarization": [
        "xsum",
        "cnn_dailymail",
    ],
    "translation": [
        "wmt_14",
        "wmt_16",
    ],
}

# Default metrics for each category
CATEGORY_METRICS = {
    "knowledge": ["accuracy", "calibration"],
    "reasoning": ["accuracy", "calibration"],
    "math": ["accuracy", "exact_match"],
    "qa": ["f1", "exact_match"],
    "truthfulness": ["accuracy", "informative"],
    "code": ["pass@1", "pass@10"],
    "bias": ["accuracy", "demographic_parity"],
    "toxicity": ["toxicity_score", "avg_toxicity"],
    "summarization": ["rouge_l", "rouge_2"],
    "translation": ["bleu", "comet"],
}


def check_helm_installed() -> bool:
    """Check if HELM is installed."""
    try:
        import helm
        return True
    except ImportError:
        console.print("[red]HELM not installed.[/red]")
        console.print("Install with: pip install crfm-helm")
        console.print("Or: pip install git+https://github.com/stanford-crfm/helm.git")
        return False


def list_scenarios():
    """List all available HELM scenarios."""
    console.print("[bold]Available HELM Scenarios[/bold]\n")

    for category, scenarios in HELM_SCENARIOS.items():
        console.print(f"[cyan]{category.upper()}:[/cyan]")
        metrics = CATEGORY_METRICS.get(category, ["accuracy"])
        for scenario in scenarios:
            console.print(f"  - {scenario} (metrics: {', '.join(metrics)})")
        console.print()


def create_helm_config(
    model_path: str,
    scenarios: List[str],
    output_dir: str,
    max_instances: Optional[int] = None,
    num_trials: int = 1,
) -> str:
    """Create HELM run configuration."""
    config = {
        "run_specs": [],
        "output_path": output_dir,
    }

    for scenario in scenarios:
        run_spec = {
            "name": f"{scenario}",
            "scenario_spec": {
                "class_name": f"helm.benchmark.scenarios.{scenario}_scenario.{scenario.title()}Scenario",
            },
            "adapter_spec": {
                "method": "generation",
                "model": model_path,
                "max_tokens": 512,
                "temperature": 0.0,
            },
            "metric_specs": [
                {"class_name": "helm.benchmark.metrics.basic_metrics.BasicMetric"},
            ],
        }

        if max_instances:
            run_spec["max_instances"] = max_instances

        config["run_specs"].append(run_spec)

    config_path = Path(output_dir) / "helm_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return str(config_path)


def run_helm_evaluation(
    model_path: str,
    scenarios: List[str],
    output_dir: str,
    max_instances: Optional[int] = None,
    num_threads: int = 4,
    use_cache: bool = True,
) -> Dict:
    """Run HELM evaluation using CLI."""
    console.print(f"[bold green]Running HELM Evaluation[/bold green]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Scenarios: {', '.join(scenarios)}")

    # Build HELM command
    cmd = [
        sys.executable, "-m", "helm.benchmark.run",
        "--conf-paths", "helm_config.json",
        "--output-path", output_dir,
        "--suite", "custom",
        "--num-threads", str(num_threads),
    ]

    if max_instances:
        cmd.extend(["--max-eval-instances", str(max_instances)])

    if use_cache:
        cmd.extend(["--cache-dir", str(Path(output_dir) / "cache")])

    # Create config
    config_path = create_helm_config(model_path, scenarios, output_dir, max_instances)

    console.print(f"\n[dim]Command: {' '.join(cmd)}[/dim]\n")

    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(output_dir).parent),
        )

        if process.returncode != 0:
            console.print(f"[red]HELM evaluation failed:[/red]")
            console.print(process.stderr)
            return {"error": process.stderr}

        # Parse results
        results = parse_helm_results(output_dir)
        return results

    except Exception as e:
        console.print(f"[red]Error running HELM: {e}[/red]")
        return {"error": str(e)}


def run_helm_lite(
    model_path: str,
    scenarios: List[str],
    output_dir: str,
    max_samples: int = 100,
) -> Dict:
    """
    Run lightweight HELM-style evaluation without full HELM installation.
    Uses lm-evaluation-harness with HELM-equivalent tasks.
    """
    console.print("[bold]Running HELM-Lite Evaluation[/bold]")
    console.print("(Using lm-evaluation-harness with HELM-equivalent tasks)")

    # Map HELM scenarios to lm-eval tasks
    helm_to_lmeval = {
        "mmlu": "mmlu",
        "gsm": "gsm8k",
        "math": "minerva_math",
        "hellaswag": "hellaswag",
        "truthfulqa": "truthfulqa_mc",
        "humaneval": "humaneval",
        "winogrande": "winogrande",
        "piqa": "piqa",
        "boolq": "boolq",
        "openbookqa": "openbookqa",
    }

    # Convert scenarios
    lm_eval_tasks = []
    for scenario in scenarios:
        if scenario in helm_to_lmeval:
            lm_eval_tasks.append(helm_to_lmeval[scenario])
        else:
            console.print(f"[yellow]Warning: No lm-eval equivalent for {scenario}[/yellow]")

    if not lm_eval_tasks:
        console.print("[red]No valid tasks to evaluate[/red]")
        return {}

    # Run lm-evaluation-harness
    try:
        from lm_eval import evaluator

        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},dtype=auto,trust_remote_code=True",
            tasks=lm_eval_tasks,
            limit=max_samples,
            batch_size="auto",
        )

        return results

    except ImportError:
        console.print("[red]lm-eval not installed. Run: pip install lm-eval[/red]")
        return {}


def parse_helm_results(output_dir: str) -> Dict:
    """Parse HELM output results."""
    results_file = Path(output_dir) / "runs" / "results.json"

    if not results_file.exists():
        # Try alternate location
        for f in Path(output_dir).rglob("results.json"):
            results_file = f
            break

    if not results_file.exists():
        return {"error": "Results file not found"}

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def display_results(results: Dict):
    """Display evaluation results in a table."""
    if "error" in results:
        console.print(f"[red]Error: {results['error']}[/red]")
        return

    table = Table(title="HELM Evaluation Results")
    table.add_column("Scenario", style="cyan")
    table.add_column("Metric", style="green")
    table.add_column("Score", style="yellow")

    if "results" in results:
        for task_name, task_results in results["results"].items():
            for metric_name, value in task_results.items():
                if isinstance(value, (int, float)) and not metric_name.endswith("_stderr"):
                    table.add_row(
                        task_name,
                        metric_name,
                        f"{value:.4f}" if isinstance(value, float) else str(value)
                    )

    console.print(table)


def log_to_wandb(results: Dict, project: str = "helm-evaluation"):
    """Log results to Weights & Biases."""
    try:
        import wandb

        wandb.init(
            project=project,
            name=f"helm-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            tags=["helm", "evaluation"],
        )

        if "results" in results:
            metrics = {}
            for task_name, task_results in results["results"].items():
                for metric_name, value in task_results.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{task_name}/{metric_name}"] = value
            wandb.log(metrics)

        wandb.finish()
        console.print("[green]Results logged to Wandb[/green]")
    except ImportError:
        logger.warning("wandb not installed")
    except Exception as e:
        logger.warning(f"Failed to log to wandb: {e}")


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results["metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "framework": "HELM",
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]Results saved to: {output_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="HELM Evaluation Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available scenarios
    python scripts/gpu/evaluate_helm.py --list-scenarios

    # Run HELM evaluation
    python scripts/gpu/evaluate_helm.py --model outputs/gpu/checkpoints/sft/final \\
        --scenarios mmlu,gsm --output outputs/eval/helm

    # Run HELM-Lite (uses lm-eval)
    python scripts/gpu/evaluate_helm.py --model Qwen/Qwen2.5-7B-Instruct \\
        --scenarios mmlu,hellaswag --lite --max-samples 100

    # Run with Wandb logging
    python scripts/gpu/evaluate_helm.py --model Qwen/Qwen2.5-7B-Instruct \\
        --scenarios truthfulqa --wandb
        """
    )

    parser.add_argument("--model", "-m", type=str, help="Model path or HuggingFace ID")
    parser.add_argument("--scenarios", "-s", type=str, help="Comma-separated scenarios")
    parser.add_argument("--output", "-o", type=str, default="outputs/eval/helm", help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Max samples per scenario")
    parser.add_argument("--num-threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--lite", action="store_true", help="Use HELM-Lite (lm-eval backend)")
    parser.add_argument("--list-scenarios", action="store_true", help="List available scenarios")
    parser.add_argument("--wandb", action="store_true", help="Log to Wandb")
    parser.add_argument("--wandb-project", type=str, default="helm-evaluation")

    args = parser.parse_args()

    if args.list_scenarios:
        list_scenarios()
        return

    if not args.model:
        console.print("[red]--model required[/red]")
        sys.exit(1)

    if not args.scenarios:
        console.print("[red]--scenarios required (e.g., mmlu,gsm,truthfulqa)[/red]")
        sys.exit(1)

    scenarios = [s.strip() for s in args.scenarios.split(",")]

    if args.lite:
        results = run_helm_lite(
            args.model,
            scenarios,
            args.output,
            max_samples=args.max_samples or 100,
        )
    else:
        if not check_helm_installed():
            console.print("[yellow]Falling back to HELM-Lite mode[/yellow]")
            results = run_helm_lite(
                args.model,
                scenarios,
                args.output,
                max_samples=args.max_samples or 100,
            )
        else:
            results = run_helm_evaluation(
                args.model,
                scenarios,
                args.output,
                max_instances=args.max_samples,
                num_threads=args.num_threads,
            )

    display_results(results)
    save_results(results, f"{args.output}/results.json")

    if args.wandb:
        log_to_wandb(results, project=args.wandb_project)


if __name__ == "__main__":
    main()
