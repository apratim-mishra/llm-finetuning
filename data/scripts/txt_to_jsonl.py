#!/usr/bin/env python3
"""
Convert a plain-text file (one prompt per line) into JSONL with chat-style messages.

Usage:
    python data/scripts/txt_to_jsonl.py --input data/tests/prompts1.txt --output data/tests/prompts1.jsonl
"""

import argparse
import json
from pathlib import Path


def convert_txt_to_jsonl(input_path: Path, output_path: Path) -> None:
    """Convert each non-empty line into a JSON object with messages -> user content."""
    lines = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not lines:
        raise ValueError(f"No non-empty lines found in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for line in lines:
            record = {"messages": [{"role": "user", "content": line}]}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert TXT prompts to JSONL messages")
    parser.add_argument("--input", "-i", required=True, help="Path to input .txt file (one prompt per line)")
    parser.add_argument("--output", "-o", help="Path to output .jsonl file (defaults to input name + .jsonl)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".jsonl")

    convert_txt_to_jsonl(input_path, output_path)
    print(f"Wrote {output_path} ({output_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
