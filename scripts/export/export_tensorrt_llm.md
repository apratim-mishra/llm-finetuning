# TensorRT-LLM Export / Build Runbook (NVIDIA GPUs)

This repo primarily targets production inference via `vLLM` (throughput) and `SGLang` (low-latency).
TensorRT-LLM is optional and only worth adding once you:
- have stable merged checkpoints you want to deploy for a long time
- are bottlenecked on latency/cost and can own the build toolchain

## Recommended pipeline

1) Train adapters (SFT/DPO/GRPO) using the GPU scripts under `scripts/gpu/`.
2) Merge the final adapter into the base model:
   - `python scripts/gpu/merge_lora.py --adapter outputs/gpu/checkpoints/<stage>/final --output outputs/gpu/merged_models/<name>`
3) Use the merged model directory as the **single source** for deployment/export.

## TensorRT-LLM build overview

Typical TensorRT-LLM workflow is:
- Convert HF weights → TensorRT-LLM compatible format (varies by model family)
- Build an engine for your target GPU + max sequence length
- Serve via a TensorRT-LLM runtime/server

Because TensorRT-LLM is moving quickly and the exact commands differ by:
- model family (Qwen/Llama/Mistral/etc.)
- TensorRT-LLM version
- CUDA/cuDNN/TensorRT versions
- target GPU (A10/A100/H100) and intended context length

this repo does not hardcode a single command. Instead:
- keep your TensorRT-LLM build steps in a separate ops repo or under `scripts/export/`
- store the engine artifacts outside git (e.g. in `outputs/` or object storage)

## Practical guidance

### When to prefer vLLM / SGLang
- Fast iteration, minimal ops friction
- You need OpenAI-compatible APIs quickly
- You want tensor-parallel serving without building engines

### When to consider TensorRT-LLM
- Stable model with long-lived deployment
- Hard latency targets on NVIDIA
- You can build per-GPU engines and manage compatibility

## Suggested repo additions (if you adopt TensorRT-LLM)

- Add a dedicated `configs/export/tensorrt_llm.yaml` with:
  - engine build parameters (dtype, max seq len, batch sizes)
  - target GPU architecture assumptions
- Add a script wrapper `scripts/export/build_tensorrt_llm.sh` that:
  - validates environment (CUDA/TensorRT versions)
  - builds the engine from `outputs/gpu/merged_models/<name>`
  - writes engine artifacts to `outputs/gpu/tensorrt_llm/<name>/`

