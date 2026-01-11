#!/usr/bin/env python3
"""
MLX Evaluation Script
Evaluate fine-tuned models on translation (BLEU) and math (accuracy) tasks.
"""

import argparse
import json
import re
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()

load = None  # type: ignore
generate = None  # type: ignore


def load_model_and_tokenizer(model_path: str, adapter_path: str = None):
    """Load MLX model with optional LoRA adapters."""
    global load, generate
    if load is None and generate is None:
        try:
            from mlx_lm import generate as mlx_generate, load as mlx_load  # type: ignore
        except ImportError:
            console.print("[red]Error: mlx-lm not installed. Run: pip install mlx-lm[/red]")
            raise

        load = mlx_load
        generate = mlx_generate

    if adapter_path and Path(adapter_path).exists():
        console.print(f"[blue]Loading model with adapters from {adapter_path}[/blue]")
        model, tokenizer = load(model_path, adapter_path=adapter_path)  # type: ignore[misc]
    else:
        console.print(f"[blue]Loading base model: {model_path}[/blue]")
        model, tokenizer = load(model_path)  # type: ignore[misc]

    return model, tokenizer, generate


def generate_response(
    model,
    tokenizer,
    generate_fn,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Generate response using MLX model."""
    response = generate_fn(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
    )
    return response


# =============================================================================
# Translation Evaluation (Use Case 1)
# =============================================================================

def evaluate_translation(
    model,
    tokenizer,
    generate_fn,
    test_file: str,
    max_samples: int = 100,
    system_prompt: str = "You are a professional Korean-English translator. Translate the given Korean text to natural, fluent English.",
) -> dict:
    """Evaluate translation model using BLEU score."""
    
    console.print("\n[bold]═══ Translation Evaluation ═══[/bold]")
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    test_data = test_data[:max_samples]
    console.print(f"Evaluating on {len(test_data)} samples...")
    
    predictions = []
    references = []
    
    for sample in track(test_data, description="Generating translations"):
        # Extract Korean source
        messages = sample.get("messages", [])
        korean_text = None
        english_ref = None
        
        for msg in messages:
            if msg["role"] == "user":
                korean_text = msg["content"]
            elif msg["role"] == "assistant":
                english_ref = msg["content"]
        
        if not korean_text or not english_ref:
            continue
        
        # Format prompt
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{korean_text}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate translation
        prediction = generate_response(
            model, tokenizer, generate_fn,
            prompt,
            max_tokens=256,
            temperature=0.1,
        )
        
        # Clean prediction
        prediction = prediction.strip()
        if "<|im_end|>" in prediction:
            prediction = prediction.split("<|im_end|>")[0].strip()
        
        predictions.append(prediction)
        references.append([english_ref])  # sacrebleu expects list of references
    
    # Calculate BLEU
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        result = bleu.corpus_score(predictions, references)
        
        console.print(f"\n[green]BLEU Score: {result.score:.2f}[/green]")
        console.print(f"Details: {result}")
        
        return {
            "bleu": result.score,
            "bleu_details": str(result),
            "num_samples": len(predictions),
        }
    except ImportError:
        console.print("[yellow]sacrebleu not installed. Install with: pip install sacrebleu[/yellow]")
        return {"error": "sacrebleu not installed"}


# =============================================================================
# Math Evaluation (Use Case 3)
# =============================================================================

def extract_answer(text: str) -> str:
    """Extract final numerical answer from solution text."""
    patterns = [
        r"Answer:\s*\$?([0-9.,\-]+)",
        r"answer is\s*\$?([0-9.,\-]+)",
        r"####\s*\$?([0-9.,\-]+)",
        r"=\s*\$?([0-9.,\-]+)\s*$",
        r"result is\s*\$?([0-9.,\-]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Clean the number
            num = match.group(1).replace(",", "").replace("$", "").rstrip(".,;:")
            return num
    
    # Fallback: last number in text
    numbers = re.findall(r"[0-9]+\.?[0-9]*", text)
    return numbers[-1].rstrip(".,;:") if numbers else ""


def normalize_answer(answer: str) -> float:
    """Normalize answer for comparison."""
    try:
        # Remove commas and convert
        clean = answer.replace(",", "").replace("$", "").strip()
        return float(clean)
    except (ValueError, AttributeError):
        return None


def evaluate_math(
    model,
    tokenizer,
    generate_fn,
    test_file: str,
    max_samples: int = 100,
    system_prompt: str = "You are a mathematical reasoning assistant. Solve problems step-by-step, showing all your work. Always end with 'Answer: [number]'.",
) -> dict:
    """Evaluate math model on GSM8K-style problems."""
    
    console.print("\n[bold]═══ Math Reasoning Evaluation ═══[/bold]")
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    test_data = test_data[:max_samples]
    console.print(f"Evaluating on {len(test_data)} samples...")
    
    correct = 0
    total = 0
    results = []
    
    for sample in track(test_data, description="Solving problems"):
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")
        
        if not question or not ground_truth:
            continue
        
        # Format prompt
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate solution
        solution = generate_response(
            model, tokenizer, generate_fn,
            prompt,
            max_tokens=512,
            temperature=0.1,
        )
        
        # Extract and compare answers
        predicted_answer = extract_answer(solution)
        pred_normalized = normalize_answer(predicted_answer)
        true_normalized = normalize_answer(ground_truth)
        
        is_correct = False
        if pred_normalized is not None and true_normalized is not None:
            # Allow small floating point difference
            is_correct = abs(pred_normalized - true_normalized) < 0.01
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question[:100] + "...",
            "predicted": predicted_answer,
            "ground_truth": ground_truth,
            "correct": is_correct,
        })
    
    accuracy = correct / total if total > 0 else 0
    
    console.print(f"\n[green]Accuracy: {accuracy:.2%} ({correct}/{total})[/green]")
    
    # Show some examples
    console.print("\n[bold]Sample Results:[/bold]")
    table = Table()
    table.add_column("Question", width=40)
    table.add_column("Predicted")
    table.add_column("Ground Truth")
    table.add_column("Correct")
    
    for r in results[:5]:
        table.add_row(
            r["question"],
            r["predicted"],
            r["ground_truth"],
            "✓" if r["correct"] else "✗"
        )
    console.print(table)
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results[:20],  # Save some examples
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MLX Model Evaluation")
    parser.add_argument(
        "--task", "-t",
        choices=["translation", "math", "both"],
        required=True,
        help="Evaluation task"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        help="Base model path"
    )
    parser.add_argument(
        "--adapter", "-a",
        type=str,
        default=None,
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=100,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/mlx/logs/eval_results.json",
        help="Output file for results"
    )
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, generate_fn = load_model_and_tokenizer(
        args.model,
        args.adapter
    )
    
    results = {}
    
    # Run evaluations
    if args.task in ["translation", "both"]:
        test_file = args.test_file or "data/processed/korean_english/test.jsonl"
        if Path(test_file).exists():
            results["translation"] = evaluate_translation(
                model, tokenizer, generate_fn,
                test_file,
                max_samples=args.max_samples,
            )
        else:
            console.print(f"[yellow]Translation test file not found: {test_file}[/yellow]")
    
    if args.task in ["math", "both"]:
        test_file = args.test_file or "data/processed/math/gsm8k_test.jsonl"
        if Path(test_file).exists():
            results["math"] = evaluate_math(
                model, tokenizer, generate_fn,
                test_file,
                max_samples=args.max_samples,
            )
        else:
            console.print(f"[yellow]Math test file not found: {test_file}[/yellow]")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    console.print(f"\n[green]Results saved to: {output_path}[/green]")


if __name__ == "__main__":
    main()
