#!/usr/bin/env python3
"""
Math Reasoning Data Preparation (Use Case 3)
Prepares data for SFT, DPO, and GRPO training stages.
"""

import json
import os
import re
from pathlib import Path

import yaml
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()


# =============================================================================
# Configuration
# =============================================================================

DATA_CONFIG = {
    "output_dir": "data/processed/math",
    "sft_train_size": 50000,    # Subset for Mac (full: 262k+)
    "sft_val_size": 2000,
    "preference_size": 5000,     # For DPO
    "grpo_prompts_size": 2000,   # For GRPO
    "max_seq_length": 2048,
    "seed": 42,
}

# System prompt for math reasoning
SYSTEM_PROMPT = """You are a mathematical reasoning assistant. Solve problems step-by-step, showing all your work clearly. 
Always end your solution with "Answer: [final numerical answer]"."""


# =============================================================================
# Data Sources
# =============================================================================

def load_mathinstruct(max_samples: int = None) -> list[dict]:
    """Load MathInstruct dataset (chain-of-thought math)."""
    console.print("[bold blue]Loading MathInstruct dataset...[/bold blue]")
    
    try:
        ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
        if max_samples:
            ds = ds.shuffle(seed=DATA_CONFIG["seed"]).select(range(min(max_samples, len(ds))))
        
        samples = []
        for item in track(ds, description="Processing MathInstruct"):
            samples.append({
                "question": item["instruction"],
                "solution": item["output"],
                "source": "mathinstruct"
            })
        return samples
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load MathInstruct: {e}[/yellow]")
        return []


def load_gsm8k(split: str = "train") -> list[dict]:
    """Load GSM8K dataset (grade school math)."""
    console.print(f"[bold blue]Loading GSM8K {split} dataset...[/bold blue]")
    
    try:
        ds = load_dataset("openai/gsm8k", "main", split=split)
        
        samples = []
        for item in track(ds, description=f"Processing GSM8K {split}"):
            # Extract final answer from GSM8K format
            answer = item["answer"].split("####")[-1].strip()
            
            samples.append({
                "question": item["question"],
                "solution": item["answer"],
                "final_answer": answer,
                "source": "gsm8k"
            })
        return samples
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load GSM8K: {e}[/yellow]")
        return []


def load_metamathqa(max_samples: int = None) -> list[dict]:
    """Load MetaMathQA dataset (augmented GSM8K)."""
    console.print("[bold blue]Loading MetaMathQA dataset...[/bold blue]")
    
    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train")
        if max_samples:
            ds = ds.shuffle(seed=DATA_CONFIG["seed"]).select(range(min(max_samples, len(ds))))
        
        samples = []
        for item in track(ds, description="Processing MetaMathQA"):
            samples.append({
                "question": item["query"],
                "solution": item["response"],
                "source": "metamathqa"
            })
        return samples
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load MetaMathQA: {e}[/yellow]")
        return []


# =============================================================================
# Data Formatting
# =============================================================================

def format_sft_sample(sample: dict) -> dict:
    """Format for SFT training (ChatML format)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["solution"]}
    ]
    
    return {
        "messages": messages,
        "source": sample.get("source", "unknown")
    }


def format_preference_pair(prompt: str, chosen: str, rejected: str) -> dict:
    """Format for DPO training (preference pairs)."""
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


def format_grpo_prompt(sample: dict) -> dict:
    """Format for GRPO training (prompts with ground truth for reward)."""
    return {
        "prompt": f"{SYSTEM_PROMPT}\n\nProblem: {sample['question']}",
        "ground_truth_answer": sample.get("final_answer", extract_answer(sample["solution"]))
    }


# =============================================================================
# Utility Functions
# =============================================================================

def extract_answer(solution: str) -> str:
    """Extract final numerical answer from solution."""
    # Try common patterns
    patterns = [
        r"Answer:\s*([0-9.,\-]+)",
        r"####\s*([0-9.,\-]+)",
        r"= ([0-9.,\-]+)$",
        r"answer is ([0-9.,\-]+)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Fallback: last number in text
    numbers = re.findall(r"[0-9.,\-]+", solution)
    return numbers[-1] if numbers else ""


def filter_math_sample(sample: dict) -> bool:
    """Filter low-quality math samples."""
    question = sample.get("question", "")
    solution = sample.get("solution", "")
    
    # Basic length checks
    if len(question) < 10 or len(solution) < 20:
        return False
    
    # Should contain some numbers
    if not re.search(r"\d", question):
        return False
    
    # Solution should have reasoning steps
    if len(solution) < 50:
        return False
    
    return True


# =============================================================================
# Preference Pair Generation (Synthetic)
# =============================================================================

def create_synthetic_preference_pairs(samples: list[dict], n_pairs: int) -> list[dict]:
    """
    Create synthetic preference pairs for DPO.
    In practice, you'd generate multiple solutions and compare.
    Here we create simple variations for demonstration.
    """
    console.print("[bold blue]Creating synthetic preference pairs...[/bold blue]")
    
    pairs = []
    import random
    random.seed(DATA_CONFIG["seed"])
    
    for sample in track(samples[:n_pairs], description="Creating pairs"):
        question = sample["question"]
        correct_solution = sample["solution"]
        
        # Create a "rejected" response (incomplete or wrong)
        # In practice, generate with model and evaluate
        rejected_solution = create_rejected_solution(correct_solution)
        
        pairs.append(format_preference_pair(
            prompt=f"{SYSTEM_PROMPT}\n\nProblem: {question}",
            chosen=correct_solution,
            rejected=rejected_solution
        ))
    
    return pairs


def create_rejected_solution(correct_solution: str) -> str:
    """Create a plausible but incorrect solution."""
    # Simple strategy: truncate or modify
    lines = correct_solution.split('\n')
    
    if len(lines) > 3:
        # Truncate (incomplete reasoning)
        return '\n'.join(lines[:len(lines)//2]) + "\n\nAnswer: I'm not sure."
    else:
        # Add wrong conclusion
        return correct_solution.replace("Answer:", "Answer: 42\n\nWait, let me recalculate... Answer:")


# =============================================================================
# Main Pipeline
# =============================================================================

def prepare_math_data(use_subset: bool = True):
    """Main data preparation pipeline for math reasoning."""
    
    console.print("[bold green]═══ Math Reasoning Data Preparation ═══[/bold green]")
    
    # Create output directory
    output_dir = Path(DATA_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data from multiple sources
    all_samples = []
    
    # MathInstruct (primary)
    mathinstruct = load_mathinstruct(
        max_samples=DATA_CONFIG["sft_train_size"] if use_subset else None
    )
    all_samples.extend(mathinstruct)
    
    # GSM8K (evaluation benchmark - also use for training)
    gsm8k_train = load_gsm8k("train")
    all_samples.extend(gsm8k_train)
    
    # MetaMathQA (augmented, optional)
    if not use_subset:
        metamath = load_metamathqa(max_samples=50000)
        all_samples.extend(metamath)
    
    console.print(f"[cyan]Total raw samples: {len(all_samples)}[/cyan]")
    
    # Filter samples
    console.print("[bold blue]Filtering samples...[/bold blue]")
    filtered_samples = [s for s in track(all_samples, description="Filtering") if filter_math_sample(s)]
    console.print(f"[cyan]After filtering: {len(filtered_samples)}[/cyan]")
    
    # Shuffle and split
    import random
    random.seed(DATA_CONFIG["seed"])
    random.shuffle(filtered_samples)
    
    # SFT splits
    sft_train = filtered_samples[:DATA_CONFIG["sft_train_size"]]
    sft_val = filtered_samples[DATA_CONFIG["sft_train_size"]:DATA_CONFIG["sft_train_size"] + DATA_CONFIG["sft_val_size"]]
    
    # Format SFT data
    console.print("[bold blue]Formatting SFT data...[/bold blue]")
    sft_train_formatted = [format_sft_sample(s) for s in track(sft_train, description="SFT train")]
    sft_val_formatted = [format_sft_sample(s) for s in track(sft_val, description="SFT val")]
    
    # Create preference pairs for DPO
    preference_source = [s for s in filtered_samples if s.get("final_answer")][:DATA_CONFIG["preference_size"] * 2]
    preference_pairs = create_synthetic_preference_pairs(preference_source, DATA_CONFIG["preference_size"])
    
    # Split preference pairs
    pref_train = preference_pairs[:int(len(preference_pairs) * 0.9)]
    pref_val = preference_pairs[int(len(preference_pairs) * 0.9):]
    
    # Create GRPO prompts
    grpo_source = [s for s in filtered_samples if s.get("final_answer")][:DATA_CONFIG["grpo_prompts_size"]]
    grpo_prompts = [format_grpo_prompt(s) for s in track(grpo_source, description="GRPO prompts")]
    
    # Save functions
    def save_jsonl(data: list[dict], filepath: Path):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save all datasets
    console.print("[bold blue]Saving datasets...[/bold blue]")
    
    save_jsonl(sft_train_formatted, output_dir / "sft_train.jsonl")
    save_jsonl(sft_val_formatted, output_dir / "sft_val.jsonl")
    save_jsonl(pref_train, output_dir / "preference_pairs.jsonl")
    save_jsonl(pref_val, output_dir / "preference_pairs_val.jsonl")
    save_jsonl(grpo_prompts, output_dir / "grpo_prompts.jsonl")
    
    # Save GSM8K test set separately for evaluation
    gsm8k_test = load_gsm8k("test")
    gsm8k_test_formatted = [{"question": s["question"], "answer": s["final_answer"]} for s in gsm8k_test]
    save_jsonl(gsm8k_test_formatted, output_dir / "gsm8k_test.jsonl")
    
    # Save config
    config_path = output_dir / "data_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump({
            **DATA_CONFIG,
            "actual_sft_train": len(sft_train_formatted),
            "actual_sft_val": len(sft_val_formatted),
            "actual_preference_train": len(pref_train),
            "actual_preference_val": len(pref_val),
            "actual_grpo_prompts": len(grpo_prompts),
        }, f)
    
    # Print summary
    console.print("\n[bold green]═══ Data Preparation Complete ═══[/bold green]")
    console.print(f"  SFT train:        {len(sft_train_formatted)}")
    console.print(f"  SFT val:          {len(sft_val_formatted)}")
    console.print(f"  Preference train: {len(pref_train)}")
    console.print(f"  Preference val:   {len(pref_val)}")
    console.print(f"  GRPO prompts:     {len(grpo_prompts)}")
    console.print(f"  GSM8K test:       {len(gsm8k_test_formatted)}")
    console.print(f"  Output dir:       {output_dir}")
    
    # Show samples
    console.print("\n[bold]SFT Sample:[/bold]")
    console.print(json.dumps(sft_train_formatted[0], indent=2, ensure_ascii=False)[:500] + "...")
    
    console.print("\n[bold]Preference Pair Sample:[/bold]")
    console.print(json.dumps(pref_train[0], indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare math reasoning data")
    parser.add_argument("--full", action="store_true",
                        help="Use full dataset (default: subset for Mac prototyping)")
    args = parser.parse_args()
    
    prepare_math_data(use_subset=not args.full)
