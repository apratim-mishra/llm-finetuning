#!/usr/bin/env python3
"""
SGLang Inference Script
High-performance inference using SGLang with RadixAttention for prefix caching.
Used by xAI, Cursor, and recommended for low-latency production serving.

Features:
- RadixAttention: Automatic prefix caching for chat applications
- Constrained decoding: JSON schema, regex patterns
- Multi-modal support: Vision-language models
- LoRA adapter hot-swapping
- OpenAI-compatible API

Usage:
    # Batch inference
    python scripts/gpu/inference_sglang.py --model Qwen/Qwen2.5-7B-Instruct \
        --input data/processed/test.jsonl --output outputs/predictions.jsonl

    # Interactive mode
    python scripts/gpu/inference_sglang.py --model Qwen/Qwen2.5-7B-Instruct --interactive

    # Start server with RadixAttention
    python scripts/gpu/inference_sglang.py --model Qwen/Qwen2.5-7B-Instruct --serve --port 30000
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import track

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


# SGLang imports - with fallbacks for when sglang is not installed
try:
    import sglang as sgl
    from sglang import function as sgl_function
except ImportError:
    # Create fallback module and decorator when sglang not installed
    sgl = None

    def sgl_function(f):
        """Fallback decorator when sglang not installed."""
        return f


def check_sglang_installed():
    """Check if SGLang is installed."""
    if sgl is None:
        console.print("[red]SGLang not installed. Run: pip install sglang[all][/red]")
        console.print("Or for CUDA 12.1: pip install sglang[all] --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/")
        return False
    return True


def load_sglang_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    context_length: Optional[int] = None,
):
    """Load model with SGLang Runtime."""
    console.print(f"[blue]Loading model with SGLang: {model_path}[/blue]")

    runtime_kwargs = {
        "model_path": model_path,
        "tp_size": tensor_parallel_size,
        "dtype": dtype,
        "trust_remote_code": True,
    }

    if context_length:
        runtime_kwargs["context_length"] = context_length

    # SGLang supports LoRA via PEFT
    if adapter_path and Path(adapter_path).exists():
        console.print(f"[blue]Loading LoRA adapter: {adapter_path}[/blue]")
        runtime_kwargs["lora_paths"] = [adapter_path]

    runtime = sgl.Runtime(**runtime_kwargs)
    sgl.set_default_backend(runtime)

    console.print("[green]SGLang Runtime initialized[/green]")
    return runtime


def format_prompt(
    text: str,
    system_prompt: Optional[str] = None,
    format_type: str = "chatml",
) -> str:
    """Format text into chat prompt."""
    if format_type == "chatml":
        if system_prompt:
            return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    elif format_type == "llama":
        if system_prompt:
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    else:
        return text


@sgl_function
def generate_response_sgl(s, prompt: str, max_tokens: int = 512, temperature: float = 0.1):
    """SGLang generation function with automatic prefix caching."""
    if sgl is None:
        raise RuntimeError("SGLang not installed. Run: pip install sglang[all]")
    s += prompt
    s += sgl.gen("response", max_tokens=max_tokens, temperature=temperature)


@sgl_function
def generate_json_response(s, prompt: str, json_schema: dict, max_tokens: int = 512):
    """Generate JSON-constrained response."""
    if sgl is None:
        raise RuntimeError("SGLang not installed. Run: pip install sglang[all]")
    s += prompt
    s += sgl.gen("response", max_tokens=max_tokens, regex=json_schema_to_regex(json_schema))


def json_schema_to_regex(schema: dict) -> str:
    """Convert JSON schema to regex pattern for constrained decoding."""
    # Simplified - SGLang has built-in JSON schema support
    return r'\{[^}]*\}'


def batch_inference(
    runtime,
    input_file: str,
    output_file: str,
    system_prompt: Optional[str] = None,
    format_type: str = "chatml",
    max_tokens: int = 512,
    temperature: float = 0.1,
):
    """Run batch inference with SGLang's efficient batching."""
    console.print(f"[blue]Processing: {input_file}[/blue]")

    # Load input data
    inputs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                inputs.append(json.loads(line))

    console.print(f"Loaded {len(inputs)} samples")

    # Prepare prompts
    prompts = []
    for item in inputs:
        if "messages" in item:
            for msg in item["messages"]:
                if msg["role"] == "user":
                    text = msg["content"]
                    break
            else:
                text = ""
        elif "prompt" in item:
            text = item["prompt"]
        elif "question" in item:
            text = item["question"]
        else:
            text = str(item)

        prompts.append(format_prompt(text, system_prompt, format_type))

    # SGLang batch generation with RadixAttention
    console.print("[bold]Generating responses with RadixAttention...[/bold]")

    sampling_params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }

    # Use SGLang's efficient batch API
    outputs = runtime.generate(
        prompts,
        sampling_params=sampling_params,
    )

    # Save results
    results = []
    for inp, output in zip(inputs, outputs):
        generated_text = output["text"].strip()

        # Clean output
        for stop in ["<|im_end|>", "</s>", "<|eot_id|>"]:
            if stop in generated_text:
                generated_text = generated_text.split(stop)[0].strip()

        result = {
            **inp,
            "generated": generated_text,
            "usage": output.get("meta_info", {}),
        }
        results.append(result)

    # Write output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    console.print(f"[green]Results saved to: {output_file}[/green]")
    return results


def interactive_mode(
    runtime,
    system_prompt: Optional[str] = None,
    format_type: str = "chatml",
    max_tokens: int = 512,
    temperature: float = 0.7,
):
    """Run interactive chat with prefix caching benefits."""
    console.print("[bold green]Interactive Mode (SGLang)[/bold green]")
    console.print("RadixAttention enabled - conversation context is cached.")
    console.print("Type 'quit' or 'exit' to end.\n")

    if system_prompt:
        console.print(f"[dim]System: {system_prompt}[/dim]\n")

    conversation_history = []
    if system_prompt:
        conversation_history.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            # Build full conversation prompt
            conversation_history.append({"role": "user", "content": user_input})

            # Format for model
            if format_type == "chatml":
                prompt = ""
                for msg in conversation_history:
                    prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                prompt += "<|im_start|>assistant\n"
            else:
                prompt = format_prompt(user_input, system_prompt, format_type)

            # Generate with SGLang
            output = runtime.generate(
                prompt,
                sampling_params={
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.95,
                }
            )

            response = output["text"].strip()
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()

            conversation_history.append({"role": "assistant", "content": response})
            console.print(f"[bold green]Assistant:[/bold green] {response}\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break


def start_server(
    model_path: str,
    adapter_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 30000,
    tensor_parallel_size: int = 1,
):
    """Start SGLang OpenAI-compatible server."""
    import subprocess

    console.print(f"[bold green]Starting SGLang Server[/bold green]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Host: {host}:{port}")
    console.print(f"  RadixAttention: Enabled (automatic prefix caching)")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host,
        "--port", str(port),
        "--tp", str(tensor_parallel_size),
        "--trust-remote-code",
    ]

    if adapter_path and Path(adapter_path).exists():
        cmd.extend(["--lora-paths", adapter_path])

    console.print(f"\n[dim]Command: {' '.join(cmd)}[/dim]\n")
    console.print(f"[green]OpenAI-compatible API at: http://{host}:{port}/v1[/green]")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="SGLang Inference with RadixAttention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Batch inference
    python scripts/gpu/inference_sglang.py --model Qwen/Qwen2.5-7B-Instruct \\
        --input data/processed/test.jsonl --output outputs/predictions.jsonl

    # Interactive mode with conversation caching
    python scripts/gpu/inference_sglang.py --model Qwen/Qwen2.5-7B-Instruct --interactive

    # Start server (OpenAI-compatible)
    python scripts/gpu/inference_sglang.py --model Qwen/Qwen2.5-7B-Instruct --serve --port 30000

    # Multi-GPU with tensor parallelism
    python scripts/gpu/inference_sglang.py --model meta-llama/Llama-3.1-70B-Instruct \\
        --serve --tensor-parallel 8
        """
    )

    parser.add_argument("--model", "-m", type=str, required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--adapter", "-a", type=str, help="LoRA adapter path")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--output", "-o", type=str, default="outputs/predictions_sglang.jsonl")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--serve", action="store_true", help="Start server")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--system-prompt", type=str, help="System prompt")
    parser.add_argument("--format", type=str, default="chatml", choices=["chatml", "llama", "raw"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--context-length", type=int, help="Override context length")

    args = parser.parse_args()

    if not check_sglang_installed():
        sys.exit(1)

    if args.serve:
        start_server(
            args.model,
            adapter_path=args.adapter,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel,
        )
        return

    # Load model
    runtime = load_sglang_model(
        args.model,
        adapter_path=args.adapter,
        tensor_parallel_size=args.tensor_parallel,
        context_length=args.context_length,
    )

    try:
        if args.interactive:
            interactive_mode(
                runtime,
                system_prompt=args.system_prompt,
                format_type=args.format,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        elif args.input:
            batch_inference(
                runtime,
                args.input,
                args.output,
                system_prompt=args.system_prompt,
                format_type=args.format,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            console.print("[red]Specify --input, --interactive, or --serve[/red]")
            sys.exit(1)
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
