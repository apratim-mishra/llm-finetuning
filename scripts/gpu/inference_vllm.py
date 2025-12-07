#!/usr/bin/env python3
"""
vLLM Inference Script
High-throughput inference for fine-tuned models using vLLM.

Supports:
- Batch inference from JSONL files
- Interactive chat mode
- LoRA adapter loading
- Streaming responses
- OpenAI-compatible API server

Usage:
    # Batch inference
    python scripts/gpu/inference_vllm.py --model outputs/gpu/checkpoints/sft/final \
        --input data/processed/test.jsonl --output outputs/predictions.jsonl

    # Interactive mode
    python scripts/gpu/inference_vllm.py --model outputs/gpu/checkpoints/sft/final --interactive

    # Start API server
    python scripts/gpu/inference_vllm.py --model outputs/gpu/checkpoints/sft/final --serve --port 8000
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


def load_vllm_model(
    model_path: str,
    adapter_path: Optional[str] = None,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    max_model_len: Optional[int] = None,
    quantization: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
):
    """Load model with vLLM for high-throughput inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        console.print("[red]vLLM not installed. Run: pip install vllm[/red]")
        sys.exit(1)

    console.print(f"[blue]Loading model: {model_path}[/blue]")

    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "trust_remote_code": True,
        "gpu_memory_utilization": gpu_memory_utilization,
    }

    if max_model_len:
        llm_kwargs["max_model_len"] = max_model_len

    if quantization:
        llm_kwargs["quantization"] = quantization

    # Load LoRA adapter if provided
    if adapter_path and Path(adapter_path).exists():
        console.print(f"[blue]Loading LoRA adapter: {adapter_path}[/blue]")
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64

    llm = LLM(**llm_kwargs)

    console.print("[green]Model loaded successfully[/green]")
    return llm, SamplingParams


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
            return f"<<SYS>>\n{system_prompt}\n<</SYS>>\n[INST] {text} [/INST]"
        return f"[INST] {text} [/INST]"
    else:
        return text


def batch_inference(
    llm,
    SamplingParams,
    input_file: str,
    output_file: str,
    system_prompt: Optional[str] = None,
    format_type: str = "chatml",
    max_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.95,
    adapter_path: Optional[str] = None,
):
    """Run batch inference on JSONL file."""
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
        # Support multiple input formats
        if "messages" in item:
            # Extract user message
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
        elif "input" in item:
            text = item["input"]
        else:
            text = str(item)

        prompts.append(format_prompt(text, system_prompt, format_type))

    # Setup sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>", "</s>", "[/INST]"],
    )

    # Generate
    console.print("[bold]Generating responses...[/bold]")

    if adapter_path and Path(adapter_path).exists():
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, adapter_path)
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)

    # Save results
    results = []
    for inp, output in zip(inputs, outputs):
        generated_text = output.outputs[0].text.strip()
        result = {
            **inp,
            "generated": generated_text,
            "finish_reason": output.outputs[0].finish_reason,
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
    llm,
    SamplingParams,
    system_prompt: Optional[str] = None,
    format_type: str = "chatml",
    max_tokens: int = 512,
    temperature: float = 0.7,
    adapter_path: Optional[str] = None,
):
    """Run interactive chat session."""
    console.print("[bold green]Interactive Mode[/bold green]")
    console.print("Type 'quit' or 'exit' to end the session.\n")

    if system_prompt:
        console.print(f"[dim]System: {system_prompt}[/dim]\n")

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        stop=["<|im_end|>", "</s>", "[/INST]"],
    )

    lora_request = None
    if adapter_path and Path(adapter_path).exists():
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("adapter", 1, adapter_path)

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            prompt = format_prompt(user_input, system_prompt, format_type)

            if lora_request:
                outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate([prompt], sampling_params)

            response = outputs[0].outputs[0].text.strip()
            console.print(f"[bold green]Assistant:[/bold green] {response}\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break


def start_api_server(
    model_path: str,
    adapter_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    tensor_parallel_size: int = 1,
):
    """Start OpenAI-compatible API server."""
    import subprocess

    console.print(f"[bold green]Starting vLLM API server[/bold green]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Host: {host}:{port}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
    ]

    if adapter_path and Path(adapter_path).exists():
        cmd.extend(["--enable-lora", "--lora-modules", f"adapter={adapter_path}"])

    console.print(f"\n[dim]Command: {' '.join(cmd)}[/dim]\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Inference for Fine-tuned Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Batch inference
    python scripts/gpu/inference_vllm.py --model Qwen/Qwen2.5-7B-Instruct \\
        --adapter outputs/gpu/checkpoints/sft/final \\
        --input data/processed/test.jsonl --output outputs/predictions.jsonl

    # Interactive mode
    python scripts/gpu/inference_vllm.py --model Qwen/Qwen2.5-7B-Instruct --interactive

    # Start API server
    python scripts/gpu/inference_vllm.py --model Qwen/Qwen2.5-7B-Instruct --serve --port 8000
        """
    )

    parser.add_argument("--model", "-m", type=str, required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--adapter", "-a", type=str, help="LoRA adapter path")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file for batch inference")
    parser.add_argument("--output", "-o", type=str, default="outputs/predictions.jsonl", help="Output JSONL file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--serve", action="store_true", help="Start API server")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--system-prompt", type=str, help="System prompt for generation")
    parser.add_argument("--format", type=str, default="chatml", choices=["chatml", "llama", "raw"], help="Prompt format")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--quantization", type=str, choices=["awq", "gptq", "squeezellm"], help="Quantization method")
    parser.add_argument("--max-model-len", type=int, help="Maximum model context length")

    args = parser.parse_args()

    if args.serve:
        start_api_server(
            args.model,
            adapter_path=args.adapter,
            host=args.host,
            port=args.port,
            tensor_parallel_size=args.tensor_parallel,
        )
        return

    # Load model for inference
    llm, SamplingParams = load_vllm_model(
        args.model,
        adapter_path=args.adapter,
        tensor_parallel_size=args.tensor_parallel,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
    )

    if args.interactive:
        interactive_mode(
            llm,
            SamplingParams,
            system_prompt=args.system_prompt,
            format_type=args.format,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            adapter_path=args.adapter,
        )
    elif args.input:
        batch_inference(
            llm,
            SamplingParams,
            args.input,
            args.output,
            system_prompt=args.system_prompt,
            format_type=args.format,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            adapter_path=args.adapter,
        )
    else:
        console.print("[red]Please specify --input for batch mode, --interactive, or --serve[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
