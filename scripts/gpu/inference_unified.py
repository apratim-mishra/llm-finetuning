#!/usr/bin/env python3
"""
Unified Inference Script for Multiple Model Types and Backends

Supports:
- Model Types: Qwen, Llama, Mistral, Phi, MLX-community models
- Backends: vLLM (throughput), SGLang (latency), HuggingFace (compatibility)
- Modes: Batch inference, interactive chat, API server

Usage:
    # Qwen model with vLLM
    python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \
        --backend vllm --input data/test.jsonl --output outputs/predictions.jsonl

    # Llama model with SGLang
    python scripts/gpu/inference_unified.py --model meta-llama/Llama-3.1-8B-Instruct \
        --backend sglang --interactive

    # MLX-community model (Mac)
    python scripts/gpu/inference_unified.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --backend mlx --interactive

    # Start API server
    python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \
        --backend vllm --serve --port 8000
"""

import argparse
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from rich.console import Console
from rich.progress import track

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ModelConfig:
    """Configuration for different model families."""
    name: str
    chat_template: str  # chatml, llama, phi, mistral
    stop_tokens: List[str]
    default_system_prompt: Optional[str] = None
    supports_system: bool = True


# Model family configurations
MODEL_CONFIGS = {
    "qwen": ModelConfig(
        name="Qwen",
        chat_template="chatml",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
        default_system_prompt="You are a helpful assistant.",
        supports_system=True,
    ),
    "llama": ModelConfig(
        name="Llama",
        chat_template="llama",
        stop_tokens=["<|eot_id|>", "</s>"],
        default_system_prompt="You are a helpful, harmless, and honest assistant.",
        supports_system=True,
    ),
    "mistral": ModelConfig(
        name="Mistral",
        chat_template="mistral",
        stop_tokens=["</s>", "[/INST]"],
        default_system_prompt=None,
        supports_system=False,  # Mistral doesn't use system prompts
    ),
    "phi": ModelConfig(
        name="Phi",
        chat_template="phi",
        stop_tokens=["<|end|>", "<|endoftext|>"],
        default_system_prompt="You are a helpful AI assistant.",
        supports_system=True,
    ),
    "gemma": ModelConfig(
        name="Gemma",
        chat_template="gemma",
        stop_tokens=["<end_of_turn>", "<eos>"],
        default_system_prompt=None,
        supports_system=False,
    ),
}


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def _extract_inference_config_path(argv: List[str]) -> Optional[str]:
    for i, token in enumerate(argv):
        if token == "--inference-config" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--inference-config="):
            return token.split("=", 1)[1]
    return None


def load_inference_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Inference config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _expand_env(cfg)

    flattened: dict[str, Any] = {}
    for key, value in cfg.items():
        if isinstance(value, dict) and key in {"generation", "runtime", "serve", "io"}:
            flattened.update(value)
        else:
            flattened[key] = value

    return flattened


def detect_model_family(model_path: str) -> str:
    """Detect model family from model path/name."""
    model_lower = model_path.lower()

    if "qwen" in model_lower:
        return "qwen"
    elif "llama" in model_lower:
        return "llama"
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    elif "phi" in model_lower:
        return "phi"
    elif "gemma" in model_lower:
        return "gemma"
    else:
        # Default to ChatML format
        return "qwen"


def format_chat_prompt(
    messages: List[Dict[str, str]],
    model_family: str = "qwen",
) -> str:
    """Format messages into model-specific chat prompt."""
    config = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS["qwen"])

    if config.chat_template == "chatml":
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    elif config.chat_template == "llama":
        prompt = "<|begin_of_text|>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    elif config.chat_template == "mistral":
        prompt = "<s>"
        user_msg = ""
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
        prompt += f"[INST] {user_msg} [/INST]"

    elif config.chat_template == "phi":
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"<|system|>\n{content}<|end|>\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}<|end|>\n"
        prompt += "<|assistant|>\n"

    elif config.chat_template == "gemma":
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role != "system":  # Gemma doesn't support system
                prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"

    else:
        # Fallback to simple format
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt += "\nassistant: "

    return prompt


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        pass

    @abstractmethod
    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs):
        pass

    def shutdown(self):
        pass


class VLLMBackend(InferenceBackend):
    """vLLM backend for high-throughput inference."""

    def __init__(self):
        self.llm = None
        self.sampling_params = None

    def load_model(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
        adapter_path: Optional[str] = None,
        **kwargs
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            console.print("[red]vLLM not installed. Run: pip install vllm[/red]")
            sys.exit(1)

        console.print(f"[blue]Loading model with vLLM: {model_path}[/blue]")

        llm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "trust_remote_code": True,
            "gpu_memory_utilization": gpu_memory_utilization,
        }

        if max_model_len:
            llm_kwargs["max_model_len"] = max_model_len

        if adapter_path and Path(adapter_path).exists():
            llm_kwargs["enable_lora"] = True

        self.llm = LLM(**llm_kwargs)

        if adapter_path and Path(adapter_path).exists():
            from vllm.lora.request import LoRARequest
            self.lora_request = LoRARequest("adapter", 1, adapter_path)
        else:
            self.lora_request = None

        console.print("[green]vLLM model loaded[/green]")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self.lora_request,
        )

        return [output.outputs[0].text for output in outputs]

    def generate_stream(self, prompt: str, **kwargs):
        # vLLM streaming not implemented in simple mode
        result = self.generate([prompt], **kwargs)[0]
        yield result


class SGLangBackend(InferenceBackend):
    """SGLang backend for low-latency inference with RadixAttention."""

    def __init__(self):
        self.runtime = None

    def load_model(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        adapter_path: Optional[str] = None,
        **kwargs
    ):
        try:
            import sglang as sgl
        except ImportError:
            console.print("[red]SGLang not installed. Run: pip install sglang[all][/red]")
            sys.exit(1)

        console.print(f"[blue]Loading model with SGLang: {model_path}[/blue]")

        runtime_kwargs = {
            "model_path": model_path,
            "tp_size": tensor_parallel_size,
            "dtype": dtype,
            "trust_remote_code": True,
        }

        if max_model_len:
            runtime_kwargs["context_length"] = max_model_len

        if adapter_path and Path(adapter_path).exists():
            runtime_kwargs["lora_paths"] = [adapter_path]

        self.runtime = sgl.Runtime(**runtime_kwargs)
        sgl.set_default_backend(self.runtime)

        console.print("[green]SGLang Runtime initialized[/green]")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = self.runtime.generate(prompts, sampling_params=sampling_params)

        if isinstance(outputs, dict):
            return [outputs["text"]]
        return [o["text"] for o in outputs]

    def generate_stream(self, prompt: str, **kwargs):
        result = self.generate([prompt], **kwargs)[0]
        yield result

    def shutdown(self):
        if self.runtime:
            self.runtime.shutdown()


class HuggingFaceBackend(InferenceBackend):
    """HuggingFace Transformers backend for compatibility."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None

    def load_model(
        self,
        model_path: str,
        dtype: str = "auto",
        adapter_path: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            console.print("[red]Transformers not installed[/red]")
            sys.exit(1)

        console.print(f"[blue]Loading model with HuggingFace: {model_path}[/blue]")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model_kwargs = {
            "trust_remote_code": True,
        }

        if dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif load_in_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            console.print(f"[blue]LoRA adapter loaded: {adapter_path}[/blue]")

        self.device = next(self.model.parameters()).device
        console.print("[green]HuggingFace model loaded[/green]")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        import torch

        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            result = self.tokenizer.decode(generated, skip_special_tokens=True)
            results.append(result)

        return results

    def generate_stream(self, prompt: str, **kwargs):
        from transformers import TextIteratorStreamer
        from threading import Thread

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "streamer": streamer,
        }

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()


class MLXBackend(InferenceBackend):
    """MLX backend for Mac Apple Silicon inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load_model(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        **kwargs
    ):
        try:
            from mlx_lm import load
        except ImportError:
            console.print("[red]MLX-LM not installed. Run: pip install mlx-lm[/red]")
            sys.exit(1)

        console.print(f"[blue]Loading model with MLX: {model_path}[/blue]")

        if adapter_path and Path(adapter_path).exists():
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
            console.print(f"[blue]LoRA adapter loaded: {adapter_path}[/blue]")
        else:
            self.model, self.tokenizer = load(model_path)

        console.print("[green]MLX model loaded[/green]")

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.1,
        **kwargs
    ) -> List[str]:
        from mlx_lm import generate

        results = []
        for prompt in prompts:
            result = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
            )
            results.append(result)

        return results

    def generate_stream(self, prompt: str, **kwargs):
        from mlx_lm import generate

        result = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 512),
            temp=kwargs.get("temperature", 0.1),
        )
        yield result


def get_backend(backend_name: str) -> InferenceBackend:
    """Get inference backend by name."""
    backends = {
        "vllm": VLLMBackend,
        "sglang": SGLangBackend,
        "hf": HuggingFaceBackend,
        "huggingface": HuggingFaceBackend,
        "mlx": MLXBackend,
    }

    if backend_name.lower() not in backends:
        console.print(f"[red]Unknown backend: {backend_name}[/red]")
        console.print(f"Available: {', '.join(backends.keys())}")
        sys.exit(1)

    return backends[backend_name.lower()]()


def batch_inference(
    backend: InferenceBackend,
    input_file: str,
    output_file: str,
    model_family: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.95,
):
    """Run batch inference on JSONL file."""
    console.print(f"[blue]Processing: {input_file}[/blue]")

    config = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS["qwen"])

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
        messages = []

        # Add system prompt
        if system_prompt and config.supports_system:
            messages.append({"role": "system", "content": system_prompt})
        elif config.default_system_prompt and config.supports_system:
            messages.append({"role": "system", "content": config.default_system_prompt})

        # Extract user content
        if "messages" in item:
            for msg in item["messages"]:
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                    break
        elif "prompt" in item:
            messages.append({"role": "user", "content": item["prompt"]})
        elif "question" in item:
            messages.append({"role": "user", "content": item["question"]})
        elif "input" in item:
            messages.append({"role": "user", "content": item["input"]})
        else:
            messages.append({"role": "user", "content": str(item)})

        prompts.append(format_chat_prompt(messages, model_family))

    # Generate
    console.print("[bold]Generating responses...[/bold]")
    outputs = backend.generate(prompts, max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    # Clean outputs
    results = []
    for inp, output in zip(inputs, outputs):
        # Clean stop tokens
        for stop in config.stop_tokens:
            if stop in output:
                output = output.split(stop)[0].strip()

        result = {
            **inp,
            "generated": output.strip(),
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
    backend: InferenceBackend,
    model_family: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
):
    """Run interactive chat."""
    config = MODEL_CONFIGS.get(model_family, MODEL_CONFIGS["qwen"])

    console.print(f"[bold green]Interactive Mode ({model_family})[/bold green]")
    console.print("Type 'quit' or 'exit' to end.\n")

    conversation = []
    if system_prompt and config.supports_system:
        conversation.append({"role": "system", "content": system_prompt})
        console.print(f"[dim]System: {system_prompt}[/dim]\n")
    elif config.default_system_prompt and config.supports_system:
        conversation.append({"role": "system", "content": config.default_system_prompt})

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            conversation.append({"role": "user", "content": user_input})
            prompt = format_chat_prompt(conversation, model_family)

            response = backend.generate(
                [prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )[0]

            # Clean response
            for stop in config.stop_tokens:
                if stop in response:
                    response = response.split(stop)[0].strip()

            conversation.append({"role": "assistant", "content": response})
            console.print(f"[bold green]Assistant:[/bold green] {response}\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            break


def main():
    config_path = _extract_inference_config_path(sys.argv[1:])
    config_defaults: dict[str, Any] = {}
    if config_path:
        try:
            config_defaults = load_inference_config(config_path)
        except Exception as e:
            console.print(f"[red]Failed to load inference config: {e}[/red]")
            sys.exit(2)

    parser = argparse.ArgumentParser(
        description="Unified Inference for Multiple Model Types and Backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Qwen model with vLLM (batch)
    python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \\
        --backend vllm --input data/test.jsonl --output outputs/predictions.jsonl

    # Llama model with SGLang (interactive)
    python scripts/gpu/inference_unified.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --backend sglang --interactive

    # MLX-community model (Mac)
    python scripts/gpu/inference_unified.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --backend mlx --interactive

    # HuggingFace with 4-bit quantization
    python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \\
        --backend hf --load-4bit --interactive

    # With LoRA adapter
    python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \\
        --adapter outputs/gpu/checkpoints/sft/final --backend vllm --input test.jsonl

Model families auto-detected: qwen, llama, mistral, phi, gemma
Backends: vllm (throughput), sglang (latency), hf (compatibility), mlx (Mac)
        """
    )

    parser.add_argument(
        "--inference-config",
        type=str,
        default=config_path,
        help="YAML config to set defaults (see configs/inference/*.yaml)",
    )
    parser.add_argument("--model", "-m", type=str, default=config_defaults.get("model"), help="Model path or HF ID")
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default=config_defaults.get("backend", "vllm"),
        choices=["vllm", "sglang", "hf", "huggingface", "mlx"],
        help="Inference backend",
    )
    parser.add_argument("--adapter", "-a", type=str, default=config_defaults.get("adapter"), help="LoRA adapter path")
    parser.add_argument("--input", "-i", type=str, default=config_defaults.get("input"), help="Input JSONL file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=config_defaults.get("output", "outputs/predictions.jsonl"),
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        default=bool(config_defaults.get("interactive", False)),
        help="Interactive mode",
    )
    parser.add_argument("--no-interactive", dest="interactive", action="store_false", help="Disable interactive mode")
    parser.add_argument(
        "--serve",
        dest="serve",
        action="store_true",
        default=bool(config_defaults.get("serve", False)),
        help="Start API server",
    )
    parser.add_argument("--no-serve", dest="serve", action="store_false", help="Disable server mode")
    parser.add_argument("--port", type=int, default=int(config_defaults.get("port", 8000)), help="Server port")
    parser.add_argument("--host", type=str, default=str(config_defaults.get("host", "0.0.0.0")), help="Server host")
    parser.add_argument("--system-prompt", type=str, default=config_defaults.get("system_prompt"), help="System prompt")
    parser.add_argument(
        "--model-family",
        type=str,
        default=config_defaults.get("model_family"),
        help="Override model family detection",
    )
    parser.add_argument("--max-tokens", type=int, default=int(config_defaults.get("max_tokens", 512)))
    parser.add_argument("--temperature", type=float, default=float(config_defaults.get("temperature", 0.1)))
    parser.add_argument("--top-p", type=float, default=float(config_defaults.get("top_p", 0.95)))
    parser.add_argument("--tensor-parallel", type=int, default=int(config_defaults.get("tensor_parallel", 1)))
    parser.add_argument(
        "--dtype",
        type=str,
        default=str(config_defaults.get("dtype", "auto")),
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (backend-dependent)",
    )
    parser.add_argument("--max-model-len", type=int, default=config_defaults.get("max_model_len"))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(config_defaults.get("gpu_memory_utilization", 0.9)),
        help="vLLM GPU memory utilization target (0-1)",
    )
    parser.add_argument(
        "--load-4bit",
        dest="load_4bit",
        action="store_true",
        default=bool(config_defaults.get("load_4bit", False)),
        help="Load in 4-bit (HF only)",
    )
    parser.add_argument("--no-load-4bit", dest="load_4bit", action="store_false", help="Disable 4-bit load")
    parser.add_argument(
        "--load-8bit",
        dest="load_8bit",
        action="store_true",
        default=bool(config_defaults.get("load_8bit", False)),
        help="Load in 8-bit (HF only)",
    )
    parser.add_argument("--no-load-8bit", dest="load_8bit", action="store_false", help="Disable 8-bit load")

    args = parser.parse_args()

    if not args.model:
        console.print("[red]Missing --model (or set model in --inference-config)[/red]")
        sys.exit(2)

    # Detect model family
    model_family = args.model_family or detect_model_family(args.model)
    console.print(f"[cyan]Model family: {model_family}[/cyan]")

    # Server mode (uses native server implementations)
    if args.serve:
        if args.backend == "vllm":
            import subprocess
            cmd = [
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.model,
                "--host", args.host,
                "--port", str(args.port),
                "--trust-remote-code",
            ]
            cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel)])
            if args.dtype != "auto":
                cmd.extend(["--dtype", args.dtype])
            if args.max_model_len:
                cmd.extend(["--max-model-len", str(args.max_model_len)])
            if args.gpu_memory_utilization:
                cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
            if args.adapter:
                cmd.extend(["--enable-lora", "--lora-modules", f"default={args.adapter}"])
            subprocess.run(cmd)
        elif args.backend == "sglang":
            import subprocess
            cmd = [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", args.model,
                "--host", args.host,
                "--port", str(args.port),
                "--trust-remote-code",
            ]
            cmd.extend(["--tp-size", str(args.tensor_parallel)])
            subprocess.run(cmd)
        else:
            console.print(f"[red]Server mode not supported for {args.backend}[/red]")
        return

    # Initialize backend
    backend = get_backend(args.backend)
    backend.load_model(
        args.model,
        adapter_path=args.adapter,
        tensor_parallel_size=args.tensor_parallel,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
    )

    try:
        if args.interactive:
            interactive_mode(
                backend,
                model_family,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        elif args.input:
            batch_inference(
                backend,
                args.input,
                args.output,
                model_family,
                system_prompt=args.system_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        else:
            console.print("[red]Specify --input, --interactive, or --serve[/red]")
            sys.exit(1)
    finally:
        backend.shutdown()


if __name__ == "__main__":
    main()
