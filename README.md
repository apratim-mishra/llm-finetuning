# LLM Fine-tuning Pipeline: MLX → Cloud GPU

A complete pipeline for fine-tuning LLMs, starting with local Mac (MLX) prototyping and scaling to cloud GPU training.

## Use Cases

1. **Korean-English Translation** - SFT fine-tuning for translation
2. **Math Reasoning** - Full pipeline: SFT → DPO → GRPO

## Quick Start

### 0. Prepare Environment Variables

```bash
cp .env.example .env
# Fill in HF_TOKEN (for gated models), WANDB_*, and MLflow if used
# Add AWS/Lambda GPU keys if you plan to launch cloud instances (see Cloud GPU Credentials)
```

### 1. Setup Project Structure

```bash
chmod +x setup_project.sh
./setup_project.sh
```

### 2. Setup Environment

**Mac (MLX):**
```bash
source setup_env.sh mlx
# Or with traditional venv:
source setup_env.sh mlx --use-venv
```

**Cloud GPU:**
```bash
source setup_env.sh gpu
```

### Simple Inference Sandbox (choose env + model)

Use this when you just want to chat with a model or sanity-check prompts before training.

1) Pick environment
- Mac (MLX): `source setup_env.sh mlx` (Metal, 4-bit models)
- Cloud GPU: `source setup_env.sh gpu` (PyTorch/TRL, CUDA)

2) Pick backend + model
- Mac MLX interactive (fast startup):  
  `python scripts/gpu/inference_unified.py --backend mlx --model mlx-community/Qwen2.5-7B-Instruct-4bit --interactive --max-tokens 256 --temperature 0.2`
- GPU vLLM interactive (best throughput):  
  `python scripts/gpu/inference_unified.py --backend vllm --model Qwen/Qwen2.5-7B-Instruct --interactive --tensor-parallel 1 --max-tokens 512 --temperature 0.2`
- GPU HF fallback (when vLLM/sglang not installed):  
  `python scripts/gpu/inference_unified.py --backend hf --model Qwen/Qwen2.5-7B-Instruct --load-4bit --interactive`
- Add `--system-prompt`, `--adapter outputs/.../sft/final`, `--input data/processed/test.jsonl --output outputs/predictions.jsonl`, or `--serve --port 8000` to match your use case.

| Target | Suggested models | Notes |
|--------|------------------|-------|
| Mac (MLX) | mlx-community/Qwen2.5-7B-Instruct-4bit, mlx-community/Llama-3.2-3B-Instruct | Runs in 4-bit, best for quick prompt tests |
| Cloud GPU | Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct | Use `--tensor-parallel` for multi-GPU vLLM |

### 3. Prepare Data

```bash
# Korean-English translation
python data/scripts/prepare_translation_data.py

# Math reasoning
python data/scripts/prepare_math_data.py
```

### 4. Train (Mac/MLX)

```bash
# Translation
python scripts/mlx/train_sft.py --config configs/mlx/sft_korean_translation.yaml

# Math
python scripts/mlx/train_sft.py --config configs/mlx/sft_math_reasoning.yaml
```

### 5. Train (Cloud GPU)

```bash
# SFT
python scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml

# DPO (after SFT)
python scripts/gpu/train_dpo.py --config configs/gpu/dpo_config.yaml

# GRPO (after DPO)
python scripts/gpu/train_grpo.py --config configs/gpu/grpo_config.yaml

# With DeepSpeed (multi-GPU)
accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json \
    scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml

# With FSDP (multi-GPU)
accelerate launch --config_file configs/gpu/fsdp/accelerate_fsdp.yaml \
    scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml
```

### 6. Inference

The unified inference script supports multiple backends and model families:

```bash
# Qwen model with vLLM (high throughput)
python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm --input data/processed/test.jsonl --output outputs/predictions.jsonl

# Llama model with SGLang (low latency)
python scripts/gpu/inference_unified.py --model meta-llama/Llama-3.1-8B-Instruct \
    --backend sglang --interactive

# MLX backend for Mac Apple Silicon
python scripts/gpu/inference_unified.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --backend mlx --interactive

# HuggingFace with 4-bit quantization
python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \
    --backend hf --load-4bit --interactive

# With LoRA adapter
python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \
    --adapter outputs/gpu/checkpoints/sft/final --backend vllm --input test.jsonl

# Start API server (vLLM or SGLang)
python scripts/gpu/inference_unified.py --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm --serve --port 8000
```

**Supported Backends:**
| Backend | Best For | Features |
|---------|----------|----------|
| vllm | High throughput batch | PagedAttention, continuous batching |
| sglang | Low latency | RadixAttention, prefix caching |
| hf | Compatibility | 4-bit/8-bit quantization |
| mlx | Mac Apple Silicon | Native Metal acceleration |

**Auto-detected Model Families:** Qwen, Llama, Mistral, Phi, Gemma

### 7. Evaluate

```bash
# Mac/MLX
python scripts/mlx/evaluate.py --task translation --adapter outputs/mlx/adapters/korean_translation
python scripts/mlx/evaluate.py --task math --adapter outputs/mlx/adapters/math_reasoning

# GPU - LM Evaluation Harness
python scripts/gpu/evaluate_lm_harness.py --model outputs/gpu/checkpoints/sft/final \
    --tasks gsm8k,arc_easy --output outputs/eval/results.json

# With vLLM backend (faster)
python scripts/gpu/evaluate_lm_harness.py --model Qwen/Qwen2.5-7B-Instruct \
    --tasks gsm8k --backend vllm

# Custom evaluation
python scripts/gpu/evaluate_lm_harness.py --custom-eval math \
    --test-file outputs/predictions.jsonl
```

## Cloud GPU Credentials (.env)

Only needed when launching or connecting to external GPUs (AWS/Lambda). Copy `.env.example` and fill:

- **AWS**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `AWS_INSTANCE_TYPE` (e.g., `g5.2xlarge`), `AWS_EC2_KEY_NAME`, `AWS_EC2_SECURITY_GROUP`, `AWS_EC2_SUBNET_ID`
- **Lambda Labs**: `LAMBDA_API_KEY`, `LAMBDA_USERNAME`, `LAMBDA_REGION` (e.g., `us-east-1`), `LAMBDA_INSTANCE_TYPE` (e.g., `gpu_1x_a10`), `LAMBDA_SSH_KEY_PATH`

With these set, you can authenticate against your cloud provider and use the GPU-backed environment set up by `setup_env.sh gpu` with the same training/inference commands.

## Implementation Progress

### Completed Steps

**Step 0:** Enhanced `src/data/data_loader.py`
- `load_jsonl()`, `save_jsonl()`, `stream_jsonl()` utilities
- `load_train_val_test_data()` with automatic splits
- Format support: ChatML, instruction, Llama, Alpaca via `get_formatting_func()`
- `format_preference_pair()` for DPO training
- Dataset validation: `validate_dataset()`, `validate_messages_format()`
- `filter_by_length()` and `deduplicate_dataset()` for data cleaning

**Step 1:** Enhanced `src/evaluation/translation_metrics.py`
- `TranslationMetrics` dataclass for structured results
- `compute_sentence_bleu()` for sentence-level BLEU
- BLEU, chrF, TER, COMET metrics with error handling
- `extract_translation_from_response()` for response cleaning
- `compute_length_ratio()` for translation length analysis

**Step 2:** Enhanced `src/rewards/math_reward.py`
- `RewardResult` dataclass for structured rewards
- Answer extraction: fractions, percentages, currency, LaTeX boxed
- `answers_match()` with absolute and relative tolerance
- `get_reward_function()` for dynamic reward selection
- `evaluate_math_batch()` for comprehensive batch evaluation

**Step 3:** Improved `scripts/gpu/train_sft.py`
- Environment variable expansion, flash attention fallback
- DeepSpeed/FSDP detection for device_map handling
- `SFTConfig` from TRL, Wandb/MLflow logging, resume support

**Step 4:** Improved `scripts/gpu/train_dpo.py`
- `DPOConfig` with preference pair support
- Preference pair validation, beta/loss type configuration
- Wandb/MLflow logging, DeepSpeed/FSDP support

**Step 5:** Improved `scripts/gpu/train_grpo.py`
- `GRPOConfig` with online RL parameters
- Integration with `src/rewards/math_reward.py`
- Wandb/MLflow logging, DeepSpeed/FSDP support
- Custom reward function wrapper for TRL

**Step 6:** Added comprehensive test suite
- `tests/conftest.py`: Pytest fixtures and configuration
- `tests/test_data_loader.py`: Data loading utilities tests
- `tests/test_math_reward.py`: Reward functions tests
- `tests/test_translation_metrics.py`: Translation metrics tests
- `tests/test_mlx_training.py`: MLX script tests with mocking

**Step 7:** Added `scripts/gpu/inference_unified.py`
- Multi-backend support: vLLM (throughput), SGLang (latency), HuggingFace (compatibility), MLX (Mac)
- Auto-detection of model families: Qwen, Llama, Mistral, Phi, Gemma
- Chat template formatting per model family
- Batch inference from JSONL files
- Interactive chat mode
- LoRA adapter loading across all backends
- OpenAI-compatible API server mode (vLLM/SGLang)

**Step 8:** Added `scripts/gpu/evaluate_lm_harness.py`
- Integration with EleutherAI lm-evaluation-harness
- Standard benchmarks: GSM8K, ARC, HellaSwag, MMLU
- Custom translation/math evaluation via `src/` modules
- Results logging to Wandb/MLflow

**Step 9:** Improved `scripts/mlx/train_sft.py`
- Environment variable expansion in config
- Wandb/MLflow logging support
- Data validation using `src/data/data_loader.py`
- Automatic validation file detection

**Step 10:** Updated requirements files
- Added MLflow, pytest-cov to base requirements
- Added vLLM to GPU requirements
- Updated TRL version for GRPO support

## Project Structure

```
llm-finetuning/
├── configs/              # Training configurations
│   ├── mlx/             # Mac MLX configs
│   ├── gpu/             # Cloud GPU configs
│   │   ├── deepspeed/   # DeepSpeed ZeRO configs
│   │   └── fsdp/        # FSDP configs
│   └── evaluation/      # Eval configs
├── data/
│   ├── raw/             # Raw datasets
│   ├── processed/       # Processed JSONL files
│   └── scripts/         # Data preparation scripts
├── src/                 # Core source code
│   ├── data/            # Data loading utilities
│   ├── evaluation/      # Translation metrics
│   └── rewards/         # Math reward functions
├── scripts/
│   ├── mlx/             # Mac training scripts
│   ├── gpu/             # GPU training scripts
│   └── common/          # Shared utilities
├── tests/               # Comprehensive test suite
├── outputs/             # Checkpoints and logs
├── requirements/        # Environment requirements
└── Makefile            # Common commands
```

## Environment Differences

| Aspect | Mac (MLX) | Cloud GPU |
|--------|-----------|-----------|
| Framework | mlx, mlx-lm | torch, transformers, trl |
| Quantization | MLX 4-bit | bitsandbytes (QLoRA) |
| Trainer | mlx_lm.lora CLI | TRL (SFTTrainer, DPOTrainer, GRPOTrainer) |
| DPO/GRPO | ❌ Not supported | ✅ Full support |
| Multi-GPU | ❌ Single device | ✅ DeepSpeed, FSDP |
| Batch Size | 1-2 (memory limited) | 4-16+ |

## Mac (24GB) Capabilities

✅ **Can Do:**
- Dataset preparation & validation
- SFT LoRA training (7B 4-bit models)
- Evaluation (BLEU, accuracy)
- Inference testing

❌ **Cannot Do:**
- DPO/GRPO (requires 2x memory)
- Full fine-tuning (7B+)
- Multi-GPU training
- Large batch training

## Training Pipeline

```
Mac Prototype (3-7 days)          Cloud GPU (3-5 days)
─────────────────────────────────────────────────────
     │                                 │
     ▼                                 ▼
┌──────────┐                    ┌──────────┐
│ Data Prep │                    │ SFT Full │
│ SFT LoRA  │ ──────────────────▶│ Training │
│ Validate  │                    └────┬─────┘
└──────────┘                          │
                                      ▼
                               ┌──────────┐
                               │   DPO    │
                               │ Training │
                               └────┬─────┘
                                    │
                                    ▼
                               ┌──────────┐
                               │   GRPO   │
                               │ Training │
                               └────┬─────┘
                                    │
                                    ▼
                               ┌──────────┐
                               │  Deploy  │
                               │  (vLLM)  │
                               └──────────┘
```

## Monitoring

All training scripts support Weights & Biases and MLflow logging:

```yaml
# Wandb configuration
wandb:
  enabled: true
  project: "llm-finetuning"
  run_name: "experiment-1"
  tags: ["sft", "translation"]

# MLflow configuration
mlflow:
  enabled: true
  tracking_uri: "mlruns"
  experiment_name: "llm-finetuning"
```

## Key Features

| Feature | Description |
|---------|-------------|
| **MLX Training** | LoRA fine-tuning on Mac Apple Silicon |
| **GPU Training** | SFT, DPO, GRPO with TRL |
| **Distributed** | DeepSpeed ZeRO, FSDP support |
| **Quantization** | QLoRA (4-bit), MLX 4-bit |
| **Inference** | Multi-backend: vLLM, SGLang, HuggingFace, MLX |
| **Model Support** | Qwen, Llama, Mistral, Phi, Gemma families |
| **Evaluation** | lm-evaluation-harness, custom metrics |
| **Logging** | Wandb, MLflow, TensorBoard |

## License

MIT
