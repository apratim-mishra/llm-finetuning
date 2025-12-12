#!/usr/bin/env python3
"""
Korean-English Translation Data Preparation (Use Case 1)
Downloads and processes parallel corpora into ChatML format.
Works on both Mac (MLX) and GPU environments.
"""

import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
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
    "output_dir": "data/processed/korean_english",
    "train_size": 50000,      # Subset for Mac prototyping (full: 500k+)
    "val_size": 2000,
    "test_size": 2000,
    "max_source_length": 256,
    "max_target_length": 256,
    "seed": 42,
}

# System prompt for translation task
SYSTEM_PROMPT = "You are a professional Korean-English translator. Translate the given Korean text to natural, fluent English."


# =============================================================================
# Data Sources
# =============================================================================

def load_opus100_ko_en(max_samples: int = None) -> list[dict]:
    """Load OPUS-100 Korean-English parallel corpus."""
    console.print("[bold blue]Loading OPUS-100 ko-en dataset...[/bold blue]")
    
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "ko-en", split="train")
        if max_samples:
            ds = ds.shuffle(seed=DATA_CONFIG["seed"]).select(range(min(max_samples, len(ds))))
        
        samples = []
        for item in track(ds, description="Processing OPUS-100"):
            samples.append({
                "korean": item["translation"]["ko"],
                "english": item["translation"]["en"],
                "source": "opus-100"
            })
        return samples
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load OPUS-100: {e}[/yellow]")
        return []


def load_tatoeba_ko_en(max_samples: int = None) -> list[dict]:
    """Load Tatoeba Korean-English sentences (high quality)."""
    console.print("[bold blue]Loading Tatoeba ko-en dataset...[/bold blue]")
    
    try:
        ds = load_dataset("tatoeba", lang1="ko", lang2="en", split="train")
        if max_samples:
            ds = ds.shuffle(seed=DATA_CONFIG["seed"]).select(range(min(max_samples, len(ds))))
        
        samples = []
        for item in track(ds, description="Processing Tatoeba"):
            samples.append({
                "korean": item["translation"]["ko"],
                "english": item["translation"]["en"],
                "source": "tatoeba"
            })
        return samples
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load Tatoeba: {e}[/yellow]")
        return []


# =============================================================================
# Data Formatting
# =============================================================================

def format_to_chatml(sample: dict) -> dict:
    """
    Convert translation pair to ChatML format.
    This format works with both MLX and TRL trainers.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["korean"]},
        {"role": "assistant", "content": sample["english"]}
    ]
    
    return {
        "messages": messages,
        "source": sample.get("source", "unknown")
    }


def format_to_text(sample: dict, tokenizer_template: str = "chatml") -> dict:
    """
    Format as single text field (alternative format).
    Useful for SFTTrainer with dataset_text_field.
    """
    if tokenizer_template == "chatml":
        text = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{sample["korean"]}<|im_end|>
<|im_start|>assistant
{sample["english"]}<|im_end|>"""
    else:
        # Generic instruction format
        text = f"""### System:
{SYSTEM_PROMPT}

### User:
{sample["korean"]}

### Assistant:
{sample["english"]}"""
    
    return {"text": text, "source": sample.get("source", "unknown")}


# =============================================================================
# Quality Filtering
# =============================================================================

def filter_sample(sample: dict) -> bool:
    """Filter out low-quality translation pairs."""
    korean = sample["korean"]
    english = sample["english"]
    
    # Length checks
    if len(korean) < 5 or len(english) < 5:
        return False
    if len(korean) > DATA_CONFIG["max_source_length"] * 4:  # Rough char estimate
        return False
    if len(english) > DATA_CONFIG["max_target_length"] * 4:
        return False
    
    # Empty or whitespace only
    if not korean.strip() or not english.strip():
        return False
    
    # Too similar (likely not translated)
    if korean == english:
        return False
    
    # Contains Korean in English or vice versa (basic check)
    # Korean Unicode range: AC00-D7AF (Hangul syllables)
    has_korean_in_english = any('\uac00' <= c <= '\ud7af' for c in english)
    if has_korean_in_english:
        return False
    
    return True


# =============================================================================
# Main Pipeline
# =============================================================================

def prepare_translation_data(
    output_format: str = "chatml",  # "chatml" or "text"
    use_subset: bool = True,        # True for Mac prototyping
):
    """Main data preparation pipeline."""
    
    console.print("[bold green]═══ Korean-English Translation Data Preparation ═══[/bold green]")
    
    # Create output directory
    output_dir = Path(DATA_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate sample sizes
    total_needed = DATA_CONFIG["train_size"] + DATA_CONFIG["val_size"] + DATA_CONFIG["test_size"]
    
    # Load data from multiple sources
    all_samples = []
    
    # OPUS-100 (larger, general quality)
    opus_samples = load_opus100_ko_en(max_samples=int(total_needed * 0.8) if use_subset else None)
    all_samples.extend(opus_samples)
    
    # Tatoeba (smaller, higher quality)
    tatoeba_samples = load_tatoeba_ko_en(max_samples=int(total_needed * 0.3) if use_subset else None)
    all_samples.extend(tatoeba_samples)
    
    console.print(f"[cyan]Total raw samples: {len(all_samples)}[/cyan]")
    
    # Filter samples
    console.print("[bold blue]Filtering samples...[/bold blue]")
    filtered_samples = [s for s in track(all_samples, description="Filtering") if filter_sample(s)]
    console.print(f"[cyan]After filtering: {len(filtered_samples)}[/cyan]")
    
    # Shuffle and split
    import random
    random.seed(DATA_CONFIG["seed"])
    random.shuffle(filtered_samples)
    
    train_samples = filtered_samples[:DATA_CONFIG["train_size"]]
    val_samples = filtered_samples[DATA_CONFIG["train_size"]:DATA_CONFIG["train_size"] + DATA_CONFIG["val_size"]]
    test_samples = filtered_samples[DATA_CONFIG["train_size"] + DATA_CONFIG["val_size"]:DATA_CONFIG["train_size"] + DATA_CONFIG["val_size"] + DATA_CONFIG["test_size"]]
    
    # Format samples
    console.print("[bold blue]Formatting samples...[/bold blue]")
    format_func = format_to_chatml if output_format == "chatml" else format_to_text
    
    train_formatted = [format_func(s) for s in track(train_samples, description="Formatting train")]
    val_formatted = [format_func(s) for s in track(val_samples, description="Formatting val")]
    test_formatted = [format_func(s) for s in track(test_samples, description="Formatting test")]
    
    # Save to JSONL
    def save_jsonl(data: list[dict], filepath: Path):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    console.print("[bold blue]Saving datasets...[/bold blue]")
    save_jsonl(train_formatted, output_dir / "train.jsonl")
    save_jsonl(val_formatted, output_dir / "val.jsonl")
    save_jsonl(test_formatted, output_dir / "test.jsonl")
    
    # Save config for reproducibility
    config_path = output_dir / "data_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump({
            **DATA_CONFIG,
            "output_format": output_format,
            "actual_train_size": len(train_formatted),
            "actual_val_size": len(val_formatted),
            "actual_test_size": len(test_formatted),
        }, f)

    # Write manifest (hashes + dataset stats)
    try:
        from src.data.manifest import build_files_manifest, write_manifest

        def _source_counts(items: list[dict]) -> dict[str, int]:
            return dict(Counter(s.get("source", "unknown") for s in items))

        manifest = {
            "task": "korean_english_translation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "platform": {
                "python": sys.version.split()[0],
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "config": {
                **DATA_CONFIG,
                "output_format": output_format,
                "use_subset": use_subset,
            },
            "counts": {
                "raw": len(all_samples),
                "filtered": len(filtered_samples),
                "train": len(train_formatted),
                "val": len(val_formatted),
                "test": len(test_formatted),
            },
            "sources": {
                "raw": _source_counts(all_samples),
                "filtered": _source_counts(filtered_samples),
                "train": _source_counts(train_formatted),
                "val": _source_counts(val_formatted),
                "test": _source_counts(test_formatted),
            },
            "files": build_files_manifest(
                {
                    "train.jsonl": output_dir / "train.jsonl",
                    "val.jsonl": output_dir / "val.jsonl",
                    "test.jsonl": output_dir / "test.jsonl",
                    "data_config.yaml": config_path,
                }
            ),
        }

        write_manifest(output_dir / "manifest.json", manifest)
        console.print(f"[green]Wrote manifest: {output_dir / 'manifest.json'}[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: could not write manifest.json: {e}[/yellow]")
    
    # Print summary
    console.print("\n[bold green]═══ Data Preparation Complete ═══[/bold green]")
    console.print(f"  Train samples: {len(train_formatted)}")
    console.print(f"  Val samples:   {len(val_formatted)}")
    console.print(f"  Test samples:  {len(test_formatted)}")
    console.print(f"  Output dir:    {output_dir}")
    console.print(f"  Format:        {output_format}")
    
    # Show sample
    console.print("\n[bold]Sample entry:[/bold]")
    console.print(json.dumps(train_formatted[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Korean-English translation data")
    parser.add_argument("--format", choices=["chatml", "text"], default="chatml",
                        help="Output format (chatml for messages, text for single field)")
    parser.add_argument("--full", action="store_true",
                        help="Use full dataset (default: subset for Mac prototyping)")
    args = parser.parse_args()
    
    prepare_translation_data(output_format=args.format, use_subset=not args.full)
