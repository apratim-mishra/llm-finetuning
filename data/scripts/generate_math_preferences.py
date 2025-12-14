#!/usr/bin/env python3
"""
Generate math preference pairs (prompt/chosen/rejected) for DPO training.

This script replaces the synthetic preference pairs from `prepare_math_data.py` with
model-generated candidates scored by `src/rewards/math_reward.py`.

Inputs:
- JSONL with prompts + ground truth answers (recommended): `data/processed/math/grpo_prompts.jsonl`
- OR JSONL with questions/answers: `{"question": "...", "answer": "42"}`
- OR JSONL with precomputed candidates: `{"question": "...", "answer": "42", "candidates": ["...", "..."]}`

Outputs (by default):
- `data/processed/math/preference_pairs.jsonl`
- `data/processed/math/preference_pairs_val.jsonl`

Generation Efficiency:
- vLLM uses `SamplingParams(n=k)` to generate k candidates per prompt in ONE call
- This is much faster than sequential generation (batched GPU inference)
- SGLang similarly batches n generations efficiently

Provenance Tracking:
- Each output pair includes full metadata in the "meta" field:
  - model_id, adapter, sampling_params (temperature, top_p, n, max_new_tokens)
  - reward_function, chosen_reward, rejected_reward, reward_gap
  - timestamp, seed
- This enables reproducibility and lineage tracking for DPO experiments

Reward Functions Available (--reward-function):
- accuracy: Partial credit for good reasoning even with wrong answer
- binary: Strict 0/1 for correct/incorrect
- combined: Accuracy + format + reasoning quality
- step_level/prm: Process Reward Model style (scores intermediate steps)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.progress import track

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preferences import PreferencePair, select_preference_pair
from src.rewards.math_reward import get_reward_function

console = Console()

DEFAULT_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Solve problems step-by-step, showing all your work clearly. "
    'Always end your solution with "Answer: [final numerical answer]".'
)


@dataclass(frozen=True)
class GenerationConfig:
    backend: str
    model: str
    adapter: Optional[str]
    model_family: Optional[str]
    tensor_parallel: int
    dtype: str
    max_model_len: Optional[int]
    gpu_memory_utilization: float
    max_new_tokens: int
    temperature: float
    top_p: float
    num_generations: int
    seed: int


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def save_jsonl(items: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _split_grpo_prompt(text: str) -> Tuple[Optional[str], str]:
    """
    Try to split a GRPO-style prompt into (system_prompt, user_text).

    The math prep script uses: f"{SYSTEM_PROMPT}\\n\\nProblem: {question}"
    """
    marker = "\n\nProblem:"
    if marker in text:
        system, rest = text.split(marker, 1)
        return system.strip(), ("Problem:" + rest).strip()
    return None, text.strip()


def _extract_prompt_and_gt(
    item: Dict[str, Any], system_prompt_default: str, model_family: str
) -> Tuple[str, str]:
    """
    Return (prompt_text, ground_truth_answer) where prompt_text is ready for DPO (ends at assistant start).
    """
    from scripts.gpu.inference_unified import format_chat_prompt

    gt = (
        item.get("ground_truth_answer")
        or item.get("answer")
        or item.get("final_answer")
        or ""
    )
    gt = str(gt).strip()

    if "messages" in item and isinstance(item["messages"], list):
        messages = [m for m in item["messages"] if m.get("role") in {"system", "user"}]
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt_default})
        return format_chat_prompt(messages, model_family), gt

    if "prompt" in item and isinstance(item["prompt"], str):
        inferred_system, user_text = _split_grpo_prompt(item["prompt"])
        system_prompt = inferred_system or system_prompt_default
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        return format_chat_prompt(messages, model_family), gt

    if "question" in item and isinstance(item["question"], str):
        messages = [
            {"role": "system", "content": system_prompt_default},
            {"role": "user", "content": item["question"]},
        ]
        return format_chat_prompt(messages, model_family), gt

    messages = [
        {"role": "system", "content": system_prompt_default},
        {"role": "user", "content": json.dumps(item, ensure_ascii=False)},
    ]
    return format_chat_prompt(messages, model_family), gt


def _detect_model_family(override: Optional[str], model: str) -> str:
    if override:
        return override
    from scripts.gpu.inference_unified import detect_model_family

    return detect_model_family(model)


def generate_candidates(
    prompts: List[str],
    cfg: GenerationConfig,
) -> List[List[str]]:
    if cfg.backend == "vllm":
        return generate_candidates_vllm(prompts, cfg)
    if cfg.backend == "sglang":
        return generate_candidates_sglang(prompts, cfg)
    if cfg.backend in {"hf", "huggingface"}:
        return generate_candidates_hf(prompts, cfg)
    raise ValueError(f"Unsupported backend: {cfg.backend}")


def generate_candidates_vllm(prompts: List[str], cfg: GenerationConfig) -> List[List[str]]:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        raise RuntimeError("vLLM not installed. Install GPU deps or use --backend hf.") from e

    llm_kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "tensor_parallel_size": cfg.tensor_parallel,
        "dtype": cfg.dtype,
        "trust_remote_code": True,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
    }
    if cfg.max_model_len:
        llm_kwargs["max_model_len"] = cfg.max_model_len
    if cfg.adapter:
        llm_kwargs["enable_lora"] = True

    llm = LLM(**llm_kwargs)

    lora_request = None
    if cfg.adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("adapter", 1, cfg.adapter)

    sampling_params = SamplingParams(
        n=cfg.num_generations,
        max_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    candidates: List[List[str]] = []
    for output in outputs:
        candidates.append([o.text for o in output.outputs])
    return candidates


def generate_candidates_sglang(prompts: List[str], cfg: GenerationConfig) -> List[List[str]]:
    try:
        import sglang as sgl
    except ImportError as e:
        raise RuntimeError("SGLang not installed. Install extras or use --backend hf.") from e

    runtime_kwargs: Dict[str, Any] = {
        "model_path": cfg.model,
        "tp_size": cfg.tensor_parallel,
        "dtype": cfg.dtype,
        "trust_remote_code": True,
    }
    if cfg.max_model_len:
        runtime_kwargs["context_length"] = cfg.max_model_len
    if cfg.adapter:
        runtime_kwargs["lora_paths"] = [cfg.adapter]

    runtime = sgl.Runtime(**runtime_kwargs)
    sgl.set_default_backend(runtime)

    expanded: List[str] = []
    for p in prompts:
        expanded.extend([p] * cfg.num_generations)

    outputs = runtime.generate(
        expanded,
        sampling_params={
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        },
    )

    texts: List[str] = []
    if isinstance(outputs, dict):
        texts = [outputs.get("text", "")]
    else:
        texts = [o.get("text", "") for o in outputs]

    grouped: List[List[str]] = []
    for i in range(len(prompts)):
        start = i * cfg.num_generations
        grouped.append(texts[start : start + cfg.num_generations])

    runtime.shutdown()
    return grouped


def generate_candidates_hf(prompts: List[str], cfg: GenerationConfig) -> List[List[str]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise RuntimeError("transformers/torch not installed. Install deps or use a GPU env.") from e

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if cfg.dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif cfg.dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16

    model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)

    if cfg.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, cfg.adapter, is_trainable=False)

    model.eval()

    batch = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_len = batch["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.temperature > 0,
            temperature=cfg.temperature if cfg.temperature > 0 else None,
            top_p=cfg.top_p if cfg.temperature > 0 else None,
            num_return_sequences=cfg.num_generations,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[:, input_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    grouped: List[List[str]] = []
    for i in range(len(prompts)):
        start = i * cfg.num_generations
        grouped.append(decoded[start : start + cfg.num_generations])

    return grouped


def to_dpo_json(pair: PreferencePair, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {
        "prompt": pair.prompt,
        "chosen": pair.chosen,
        "rejected": pair.rejected,
    }
    if meta:
        out["meta"] = meta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate math preference pairs for DPO")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/processed/math/grpo_prompts.jsonl",
        help="Input JSONL with prompts + ground truth answers",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/processed/math",
        help="Output directory for preference_pairs*.jsonl",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "sglang", "hf", "huggingface"],
        help="Generation backend",
    )
    parser.add_argument("--model", type=str, required=False, help="Model HF id/path")
    parser.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter path")
    parser.add_argument("--model-family", type=str, default=None, help="Override model family for prompt formatting")
    parser.add_argument("--tensor-parallel", type=int, default=1, help="Tensor parallel size (vLLM/SGLang)")
    parser.add_argument("--dtype", type=str, default="auto", help="dtype for generation backend")
    parser.add_argument("--max-model-len", type=int, default=None, help="Max context length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization")

    parser.add_argument("--num-generations", type=int, default=4, help="Candidates per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts processed")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--reward-function",
        type=str,
        default="combined",
        help="Reward function name from src/rewards/math_reward.py (accuracy|binary|format|combined)",
    )
    parser.add_argument("--min-reward-gap", type=float, default=0.0, help="Skip pairs with small reward gap")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(input_path)
    if args.limit:
        items = items[: args.limit]

    all_have_candidates = all(
        isinstance(item.get("candidates"), list) and len(item.get("candidates", [])) > 0 for item in items
    )
    if not args.model and not all_have_candidates:
        console.print("[red]Missing --model (required unless every input row has candidates)[/red]")
        console.print("Tip: pass --backend hf if vLLM/SGLang are not installed.")
        raise SystemExit(2)

    if args.model:
        model_family = _detect_model_family(args.model_family, args.model)
    else:
        model_family = args.model_family or "qwen"

    gen_cfg = GenerationConfig(
        backend=args.backend,
        model=args.model or "",
        adapter=args.adapter,
        model_family=model_family,
        tensor_parallel=args.tensor_parallel,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_generations=args.num_generations,
        seed=args.seed,
    )

    reward_base = get_reward_function(args.reward_function)
    reward_fn = lambda comps, gts: reward_base(comps, gts)  # noqa: E731

    prompts: List[str] = []
    ground_truths: List[str] = []
    candidates_from_input: List[Optional[List[str]]] = []

    for item in items:
        prompt_text, gt = _extract_prompt_and_gt(item, args.system_prompt, model_family)
        prompts.append(prompt_text)
        ground_truths.append(gt)
        cands = item.get("candidates") if isinstance(item.get("candidates"), list) else None
        candidates_from_input.append(cands)

    generated_candidates: List[Optional[List[str]]] = [None] * len(prompts)
    missing_idx = [i for i, c in enumerate(candidates_from_input) if c is None]
    if missing_idx:
        prompts_to_generate = [prompts[i] for i in missing_idx]
        console.print(f"[blue]Generating candidates for {len(prompts_to_generate)} prompts...[/blue]")
        cand_groups = generate_candidates(prompts_to_generate, gen_cfg)
        if len(cand_groups) != len(prompts_to_generate):
            raise RuntimeError("Candidate generation returned unexpected length")
        for idx, cand_list in zip(missing_idx, cand_groups):
            generated_candidates[idx] = cand_list

    pairs: List[Dict[str, Any]] = []
    skipped = 0
    for i in track(range(len(prompts)), description="Selecting preference pairs"):
        cands = candidates_from_input[i] if candidates_from_input[i] is not None else generated_candidates[i]
        if not cands:
            skipped += 1
            continue

        pair = select_preference_pair(
            prompt=prompts[i],
            candidates=cands,
            ground_truth=ground_truths[i],
            reward_fn=reward_fn,
            min_reward_gap=args.min_reward_gap,
        )
        if pair is None:
            skipped += 1
            continue

        # Full provenance metadata for reproducibility
        provenance_meta = {
            "model_id": args.model or "precomputed_candidates",
            "adapter": args.adapter,
            "sampling_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "n": args.num_generations,
                "max_new_tokens": args.max_new_tokens,
            },
            "reward_function": args.reward_function,
            "chosen_reward": pair.chosen_reward,
            "rejected_reward": pair.rejected_reward,
            "reward_gap": pair.chosen_reward - pair.rejected_reward,
            "min_reward_gap": args.min_reward_gap,
            "timestamp": datetime.now().isoformat(),
            "seed": args.seed,
        }
        pairs.append(to_dpo_json(pair, meta=provenance_meta))

    console.print(f"[green]Built {len(pairs)} preference pairs (skipped: {skipped})[/green]")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    val_size = int(len(pairs) * args.val_ratio)
    val_pairs = pairs[:val_size]
    train_pairs = pairs[val_size:]

    train_out = output_dir / "preference_pairs.jsonl"
    val_out = output_dir / "preference_pairs_val.jsonl"
    save_jsonl(train_pairs, train_out)
    save_jsonl(val_pairs, val_out)

    console.print(f"[green]Wrote train pairs: {train_out} ({len(train_pairs)})[/green]")
    console.print(f"[green]Wrote val pairs:   {val_out} ({len(val_pairs)})[/green]")


if __name__ == "__main__":
    main()
