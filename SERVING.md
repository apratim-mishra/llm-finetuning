# Serving (vLLM / SGLang)

This repo supports production-style inference via:
- **vLLM** (best throughput / batching)
- **SGLang** (best low-latency + prefix caching / structured generation)

For quick local testing on Mac, use the **MLX** backend.

## Recommended artifact strategy

You have two common deployment options:

1) **Serve base model + LoRA adapter** (fast iteration)
- Pros: no merge step, easy to swap adapters
- Cons: extra moving parts for deployment

2) **Serve a merged model** (simplest deployment)
- Pros: one model directory to ship (`outputs/gpu/merged_models/...`)
- Cons: requires a merge step and enough memory to merge

Merge LoRA:
- `python scripts/gpu/merge_lora.py --adapter outputs/gpu/checkpoints/<stage>/final --output outputs/gpu/merged_models/<name>`

## Using inference presets

The unified script supports `--inference-config`:
- `configs/inference/translation_vllm.yaml`
- `configs/inference/translation_sglang.yaml`
- `configs/inference/math_vllm.yaml`
- `configs/inference/math_sglang.yaml`
- `configs/inference/mlx_local.yaml`

You can still override any CLI flag after the config.

## vLLM (OpenAI-compatible server)

Single GPU:
- `python scripts/gpu/inference_unified.py --inference-config configs/inference/translation_vllm.yaml --serve --port 8000`

Serve with adapter:
- `python scripts/gpu/inference_unified.py --inference-config configs/inference/translation_vllm.yaml --adapter outputs/gpu/checkpoints/sft/final --serve --port 8000`

Multi-GPU tensor parallel:
- `CUDA_VISIBLE_DEVICES=0,1 python scripts/gpu/inference_unified.py --inference-config configs/inference/translation_vllm.yaml --tensor-parallel 2 --serve --port 8000`

Useful knobs:
- `--max-model-len 4096` (or higher if supported)
- `--gpu-memory-utilization 0.9` (lower if you see OOMs)
- `--dtype bfloat16` (Ampere+ recommended)

## SGLang server

Single GPU:
- `python scripts/gpu/inference_unified.py --inference-config configs/inference/translation_sglang.yaml --serve --port 30000`

Multi-GPU tensor parallel:
- `CUDA_VISIBLE_DEVICES=0,1 python scripts/gpu/inference_unified.py --inference-config configs/inference/translation_sglang.yaml --tensor-parallel 2 --serve --port 30000`

Adapter notes:
- SGLang supports LoRA paths; for deployment simplicity, prefer merged models once stable.

## Smoke-check (no server)

Batch:
- `python scripts/gpu/inference_unified.py --inference-config configs/inference/math_vllm.yaml --input data/processed/math/gsm8k_test.jsonl --output outputs/predictions.jsonl`

Interactive:
- `python scripts/gpu/inference_unified.py --inference-config configs/inference/mlx_local.yaml`

