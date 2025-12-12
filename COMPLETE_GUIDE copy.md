# Complete LLM Fine-Tuning Guide: MLX to Cloud GPU

> **A comprehensive guide for fine-tuning LLMs starting with Mac (MLX) development and scaling to cloud GPU training.**

Note: `README.md` is the canonical, most up-to-date quickstart; this guide may lag behind.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Use Cases](#2-use-cases)
3. [Technology Stack](#3-technology-stack)
4. [Hardware Requirements & Capabilities](#4-hardware-requirements--capabilities)
5. [Project Structure](#5-project-structure)
6. [Environment Setup](#6-environment-setup)
7. [Data Preparation](#7-data-preparation)
8. [Training Pipeline](#8-training-pipeline)
9. [Evaluation](#9-evaluation)
10. [Deployment](#10-deployment)
11. [Complete Workflow Timeline](#11-complete-workflow-timeline)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

### Purpose

This project provides a complete workflow for fine-tuning Large Language Models (LLMs) with a **two-phase approach**:

1. **Phase 1: Mac Development (MLX)** - Rapid prototyping, data validation, hyperparameter exploration
2. **Phase 2: Cloud GPU Training** - Full-scale training with advanced methods (DPO, GRPO)

### Why This Approach?

| Phase | Purpose | Duration | Cost |
|-------|---------|----------|------|
| Mac (MLX) | Validate data, test configs, prototype | 3-7 days | $0 |
| Cloud GPU | Production training, advanced RL | 3-5 days | $50-500 |

### Training Methods Supported

| Method | Description | Mac (MLX) | Cloud GPU |
|--------|-------------|-----------|-----------|
| **SFT** | Supervised Fine-Tuning | ✅ LoRA only | ✅ Full + LoRA |
| **DPO** | Direct Preference Optimization | ❌ | ✅ |
| **GRPO** | Group Relative Policy Optimization | ❌ | ✅ |
| **PPO** | Proximal Policy Optimization | ❌ | ✅ |

---

## 2. Use Cases

### Use Case 1: Korean-English Translation

**Goal:** Fine-tune a model to translate Korean text to fluent English.

**Datasets:**
| Dataset | Size | Source | Quality |
|---------|------|--------|---------|
| Helsinki-NLP/opus-100 (ko-en) | 1M pairs | HuggingFace | General |
| NHNDQ/nllb-translation-ko-en | 500k pairs | HuggingFace | High |
| Tatoeba (ko-en) | 100k pairs | HuggingFace | Very High |
| AI Hub Korean-English | 1M+ pairs | Korean Gov | Professional |

**Recommended Models:**
| Model | Size | MLX Available | Notes |
|-------|------|---------------|-------|
| Qwen2.5-7B-Instruct | 7B | ✅ 4-bit | Best multilingual |
| KULLM3-7B | 7B | ❌ | Korean-optimized |
| Llama-3.1-8B-Instruct | 8B | ✅ 4-bit | Strong baseline |

**Evaluation Metrics:**
- **BLEU** - N-gram overlap (primary)
- **COMET** - Semantic similarity
- **chrF** - Character-level F-score
- **TER** - Translation Edit Rate

**Expected Results:**
| Stage | BLEU Score |
|-------|------------|
| Base model | 15-25 |
| After Mac SFT | 25-35 |
| After GPU SFT | 35-45 |

---

### Use Case 3: Math Reasoning

**Goal:** Train a model to solve math problems with step-by-step reasoning.

**Datasets:**
| Dataset | Size | Type | Difficulty |
|---------|------|------|------------|
| TIGER-Lab/MathInstruct | 262k | Chain-of-thought | Mixed |
| meta-math/MetaMathQA | 395k | Augmented GSM8K | Grade school |
| openai/gsm8k | 7.5k train / 1.3k test | Grade school | Easy-Medium |
| hendrycks/math | 12.5k | Competition | Hard |
| argilla/math-preferences | 10k | Preference pairs | For DPO |

**Recommended Models:**
| Model | Size | MLX Available | Notes |
|-------|------|---------------|-------|
| Qwen2.5-Math-7B-Instruct | 7B | ✅ 4-bit | Math-specialized |
| DeepSeek-Math-7B | 7B | ❌ | Strong reasoning |
| Qwen2.5-7B-Instruct | 7B | ✅ 4-bit | General purpose |

**Evaluation Metrics:**
- **GSM8K Accuracy** - Grade school math (target: 85%+)
- **MATH Accuracy** - Competition math (target: 30%+)
- **Minerva Math** - Advanced (target: 20%+)

**Expected Results:**
| Stage | GSM8K Accuracy |
|-------|----------------|
| Base Qwen2.5-7B | 50-55% |
| After Mac SFT | 55-65% |
| After GPU SFT | 75-80% |
| After DPO | 80-85% |
| After GRPO | 85-88% |

---

## 3. Technology Stack

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DEVELOPMENT (Mac M1/M2/M3/M4)                     │
├─────────────────────────────────────────────────────────────────────────┤
│  MLX Framework │ mlx-lm │ Transformers (CPU) │ Datasets │ Wandb        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ Export: configs, data format, baselines
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING (Cloud GPU)                              │
├─────────────────────────────────────────────────────────────────────────┤
│  PyTorch │ Transformers │ TRL │ PEFT │ DeepSpeed │ FSDP │ Flash Attn   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 │ Export: merged model, adapters
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE (Production)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  vLLM │ SGLang │ TensorRT-LLM │ MLX (Mac only)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Technology Positioning

#### Training Technologies

| Technology | Purpose | When to Use |
|------------|---------|-------------|
| **MLX** | Mac Apple Silicon training | Prototyping, small models, Mac deployment |
| **TRL** | Training library (SFT, DPO, GRPO) | All GPU training |
| **PEFT** | Parameter-efficient fine-tuning | LoRA, QLoRA adapters |
| **DeepSpeed** | Distributed training | Multi-GPU, 70B+ models |
| **FSDP** | PyTorch native distributed | Multi-GPU, easier setup |
| **Accelerate** | Training orchestration | Device management, mixed precision |

#### Inference Technologies

| Technology | Purpose | Throughput | Best For |
|------------|---------|------------|----------|
| **vLLM** | Production serving | Highest | OpenAI-compatible API |
| **SGLang** | Structured generation | High (6x for JSON) | Agents, constrained output |
| **TensorRT-LLM** | NVIDIA optimized | Very High | NVIDIA GPUs only |
| **MLX** | Mac inference | Moderate | Local Mac deployment |

### Package Dependencies

#### Shared (Both Environments)
```
datasets>=2.19.0          # Data loading
pandas>=2.0.0             # Data processing
numpy>=1.24.0             # Numerical ops
tokenizers>=0.15.0        # Fast tokenization
sacrebleu>=2.4.0          # Translation metrics
evaluate>=0.4.0           # Evaluation framework
wandb>=0.16.0             # Experiment tracking
huggingface-hub>=0.21.0   # Model hub
pyyaml>=6.0.0             # Config files
rich>=13.0.0              # Pretty output
```

#### Mac (MLX)
```
mlx>=0.12.0               # Apple ML framework
mlx-lm>=0.12.0            # LLM utilities for MLX
transformers>=4.40.0      # Tokenizers, configs (CPU)
peft>=0.10.0              # LoRA config compatibility
kiwipiepy>=0.16.0         # Korean NLP
```

#### Cloud GPU
```
torch>=2.2.0              # PyTorch with CUDA
transformers>=4.40.0      # Full transformers
accelerate>=0.28.0        # Training orchestration
peft>=0.10.0              # LoRA, QLoRA
trl>=0.8.0                # SFT, DPO, GRPO trainers
bitsandbytes>=0.43.0      # 4-bit quantization
deepspeed>=0.14.0         # Distributed training
flash-attn>=2.5.0         # Flash Attention 2
lm-eval>=0.4.0            # Evaluation harness
vllm>=0.4.0               # Production inference
```

---

## 4. Hardware Requirements & Capabilities

### Mac (Apple Silicon)

| Spec | Minimum | Recommended |
|------|---------|-------------|
| Chip | M1 | M2 Pro / M3 / M4 |
| RAM | 16GB | 24GB+ |
| Storage | 50GB free | 100GB+ SSD |
| macOS | 13.0+ | 14.0+ |

**What 24GB Mac Can Do:**
- ✅ Full dataset preparation and validation
- ✅ Tokenization and data exploration
- ✅ SFT LoRA training (7B 4-bit models)
- ✅ Evaluation (BLEU, GSM8K accuracy)
- ✅ Inference (15-25 tokens/sec)
- ✅ Export configs for GPU training

**What 24GB Mac Cannot Do:**
- ❌ Full fine-tuning of 7B+ models
- ❌ DPO/GRPO (requires 2x memory for reference model)
- ❌ Large batch training (limited to 1-2)
- ❌ Multi-GPU simulation
- ❌ Training 13B+ models

### Cloud GPU Options

| Provider | GPU | VRAM | Cost/hr | Best For |
|----------|-----|------|---------|----------|
| Lambda Labs | A100 80GB | 80GB | $1.10 | Full training |
| Lambda Labs | H100 80GB | 80GB | $2.00 | Fastest training |
| AWS | A10G | 24GB | $1.00 | Budget training |
| AWS | A100 40GB | 40GB | $3.00 | Multi-GPU |
| RunPod | A100 80GB | 80GB | $1.50 | Flexible |
| Vast.ai | Various | Varies | $0.30+ | Cheapest |

**GPU Memory Requirements:**

| Model Size | SFT (QLoRA) | DPO | GRPO | Full Fine-tune |
|------------|-------------|-----|------|----------------|
| 7B | 16GB | 24GB | 32GB | 80GB |
| 13B | 24GB | 40GB | 48GB | 160GB |
| 70B | 80GB | 160GB | 200GB | 640GB |

---

## 5. Project Structure

```
llm-finetuning/
│
├── configs/                          # All configuration files
│   ├── mlx/                         # Mac MLX configs
│   │   ├── sft_korean_translation.yaml
│   │   └── sft_math_reasoning.yaml
│   ├── gpu/                         # Cloud GPU configs
│   │   ├── sft_config.yaml          # SFT training
│   │   ├── dpo_config.yaml          # DPO training
│   │   ├── grpo_config.yaml         # GRPO training
│   │   ├── deepspeed/               # DeepSpeed configs
│   │   │   ├── ds_zero2.json        # ZeRO Stage 2
│   │   │   └── ds_zero3_offload.json# ZeRO Stage 3 + CPU
│   │   └── fsdp/                    # FSDP configs
│   │       └── accelerate_fsdp.yaml
│   └── evaluation/                  # Evaluation configs
│
├── data/                            # Data directory
│   ├── raw/                         # Original downloads
│   │   ├── korean_english/
│   │   └── math/
│   ├── processed/                   # Processed JSONL files
│   │   ├── korean_english/
│   │   │   ├── train.jsonl
│   │   │   ├── val.jsonl
│   │   │   └── test.jsonl
│   │   └── math/
│   │       ├── sft_train.jsonl
│   │       ├── sft_val.jsonl
│   │       ├── preference_pairs.jsonl    # For DPO
│   │       ├── grpo_prompts.jsonl        # For GRPO
│   │       └── gsm8k_test.jsonl
│   └── scripts/                     # Data preparation
│       ├── prepare_translation_data.py
│       └── prepare_math_data.py
│
├── src/                             # Source code
│   ├── __init__.py
│   ├── data/                        # Data utilities
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── training/                    # Training utilities
│   │   └── __init__.py
│   ├── evaluation/                  # Evaluation metrics
│   │   ├── __init__.py
│   │   └── translation_metrics.py
│   ├── rewards/                     # Reward functions (GRPO)
│   │   ├── __init__.py
│   │   └── math_reward.py
│   └── inference/                   # Inference utilities
│       └── __init__.py
│
├── scripts/                         # Training scripts
│   ├── mlx/                         # Mac MLX scripts
│   │   ├── train_sft.py
│   │   └── evaluate.py
│   ├── gpu/                         # Cloud GPU scripts
│   │   ├── train_sft.py
│   │   ├── train_dpo.py
│   │   └── train_grpo.py
│   └── common/                      # Shared utilities
│
├── notebooks/                       # Jupyter notebooks
│
├── outputs/                         # Training outputs
│   ├── mlx/
│   │   ├── checkpoints/
│   │   ├── adapters/
│   │   └── logs/
│   └── gpu/
│       ├── checkpoints/
│       ├── merged_models/
│       └── logs/
│
├── tests/                           # Unit tests
│
├── requirements/                    # Dependencies
│   ├── requirements-base.txt
│   ├── requirements-mlx.txt
│   └── requirements-gpu.txt
│
├── docs/                            # Documentation
│   └── COMPLETE_GUIDE.md           # This file
│
├── pyproject.toml                   # Project config (uv)
├── setup_project.sh                 # Folder structure setup
├── setup_env.sh                     # Environment setup
├── Makefile                         # Common commands
├── README.md                        # Quick start
├── .gitignore
├── .python-version                  # Python 3.11
└── .env.example                     # Environment variables
```

---

## 6. Environment Setup

### Option A: Using uv (Recommended - Faster)

**Install uv:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

**Setup MLX Environment:**
```bash
cd llm-finetuning

# Create virtual environment
uv venv .venv-mlx --python 3.11

# Activate
source .venv-mlx/bin/activate

# Install MLX dependencies
uv pip install -r requirements/requirements-mlx.txt

# Verify
python -c "import mlx; print(f'MLX version: {mlx.__version__}')"
```

**Setup GPU Environment:**
```bash
cd llm-finetuning

# Create virtual environment
uv venv .venv-gpu --python 3.11

# Activate
source .venv-gpu/bin/activate

# Install GPU dependencies
uv pip install -r requirements/requirements-gpu.txt

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option B: Using Traditional venv

**Setup MLX Environment:**
```bash
cd llm-finetuning

# Create virtual environment
python3.11 -m venv .venv-mlx

# Activate
source .venv-mlx/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements/requirements-mlx.txt
```

**Setup GPU Environment:**
```bash
cd llm-finetuning

# Create virtual environment
python3.11 -m venv .venv-gpu

# Activate
source .venv-gpu/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies (with CUDA index)
pip install -r requirements/requirements-gpu.txt
```

### Option C: Using pyproject.toml with uv

```bash
cd llm-finetuning

# Install with MLX extras
uv pip install -e ".[mlx]"

# Or with GPU extras
uv pip install -e ".[gpu]"

# Or with everything
uv pip install -e ".[mlx,gpu,dev,eval]"
```

### Environment Variables

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env`:
```bash
# Hugging Face (required for gated models)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Weights & Biases (for experiment tracking)
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=llm-finetuning

# GPU settings
CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Verifying Installations

**Mac (MLX):**
```bash
# Check MLX
python -c "import mlx; import mlx.core as mx; print(f'MLX {mlx.__version__}')"

# Check mlx-lm
python -c "from mlx_lm import load; print('mlx-lm OK')"

# Check memory
python -c "import mlx.core as mx; print(f'Metal available: {mx.metal.is_available()}')"
```

**Cloud GPU:**
```bash
# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check Flash Attention
python -c "from flash_attn import flash_attn_func; print('Flash Attention OK')"

# Check TRL
python -c "from trl import SFTTrainer, DPOTrainer; print('TRL OK')"

# Check DeepSpeed
python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__}')"
```

---

## 7. Data Preparation

### Data Format (ChatML)

All data uses ChatML format in JSONL files:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "User input here"},
    {"role": "assistant", "content": "Model response here"}
  ]
}
```

### Prepare Translation Data

```bash
# Activate environment
source .venv-mlx/bin/activate

# Subset for Mac prototyping (50k samples)
python data/scripts/prepare_translation_data.py

# Full dataset for GPU training (500k+ samples)
python data/scripts/prepare_translation_data.py --full
```

**Output Files:**
```
data/processed/korean_english/
├── train.jsonl      # Training data
├── val.jsonl        # Validation data
├── test.jsonl       # Test data
└── data_config.yaml # Config used
```

### Prepare Math Data

```bash
# Activate environment
source .venv-mlx/bin/activate

# Subset for Mac prototyping
python data/scripts/prepare_math_data.py

# Full dataset for GPU training
python data/scripts/prepare_math_data.py --full
```

**Output Files:**
```
data/processed/math/
├── sft_train.jsonl        # SFT training
├── sft_val.jsonl          # SFT validation
├── preference_pairs.jsonl # DPO training (chosen/rejected)
├── preference_pairs_val.jsonl
├── grpo_prompts.jsonl     # GRPO training (prompts + answers)
├── gsm8k_test.jsonl       # Evaluation benchmark
└── data_config.yaml
```

### DPO Data Format

```json
{
  "prompt": "System prompt + Problem: ...",
  "chosen": "Correct step-by-step solution...",
  "rejected": "Incorrect or incomplete solution..."
}
```

### GRPO Data Format

```json
{
  "prompt": "System prompt + Problem: ...",
  "ground_truth_answer": "42"
}
```

---

## 8. Training Pipeline

### Step 0: Mac MLX Training (3-7 days)

**Purpose:** Validate data, test hyperparameters, establish baselines.

#### Translation Training

```bash
# Activate MLX environment
source .venv-mlx/bin/activate

# Train with config
python scripts/mlx/train_sft.py \
  --config configs/mlx/sft_korean_translation.yaml

# Or with overrides
python scripts/mlx/train_sft.py \
  --config configs/mlx/sft_korean_translation.yaml \
  --iters 500 \
  --batch-size 1
```

**Config Explained (`configs/mlx/sft_korean_translation.yaml`):**
```yaml
model:
  name: "mlx-community/Qwen2.5-7B-Instruct-4bit"  # 4-bit for memory
  max_seq_length: 1024                            # Shorter for translation

lora:
  rank: 32        # Lower than GPU (memory constraint)
  alpha: 64       # 2x rank
  dropout: 0.05
  layers: 16      # Apply to 16 transformer layers

training:
  iters: 1000           # Quick validation run
  batch_size: 2         # Max for 24GB
  learning_rate: 2.0e-5
  grad_checkpoint: true # Essential for memory
```

#### Math Training

```bash
python scripts/mlx/train_sft.py \
  --config configs/mlx/sft_math_reasoning.yaml
```

**Key Differences for Math:**
```yaml
model:
  max_seq_length: 2048  # Longer for reasoning chains

lora:
  rank: 64              # Higher for complex task

training:
  iters: 2000           # More iterations needed
  batch_size: 1         # Longer sequences = smaller batch
```

#### MLX Training Limitations

| Limitation | Workaround |
|------------|------------|
| No DPO/GRPO | Use GPU for preference learning |
| Batch size 1-2 | Gradient accumulation (manual) |
| No Flash Attention | Default attention OK for prototyping |
| No wandb native | Manual logging in script |

---

### Step 1: GPU SFT Training (1-2 days)

**Purpose:** Full-scale supervised fine-tuning.

#### Single GPU Training

```bash
# Activate GPU environment
source .venv-gpu/bin/activate

# Train
python scripts/gpu/train_sft.py \
  --config configs/gpu/sft_config.yaml
```

**Config Explained (`configs/gpu/sft_config.yaml`):**
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"  # 2-4x speedup
  device_map: "auto"

quantization:
  enabled: true
  load_in_4bit: true                # QLoRA
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"

lora:
  r: 64                   # Higher than Mac
  lora_alpha: 128
  target_modules: "all-linear"

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4    # Much larger than Mac
  gradient_accumulation_steps: 4    # Effective batch: 16
  learning_rate: 2.0e-5
  bf16: true
  gradient_checkpointing: true

sft:
  max_seq_length: 2048
  packing: true           # Pack multiple samples
```

#### Multi-GPU with DeepSpeed

```bash
# ZeRO Stage 2 (recommended baseline)
accelerate launch \
  --config_file configs/gpu/deepspeed/ds_zero2.json \
  scripts/gpu/train_sft.py \
  --config configs/gpu/sft_config.yaml

# ZeRO Stage 3 with CPU offload (for 70B+ models)
accelerate launch \
  --config_file configs/gpu/deepspeed/ds_zero3_offload.json \
  scripts/gpu/train_sft.py \
  --config configs/gpu/sft_config.yaml
```

**DeepSpeed ZeRO Stages:**
| Stage | What's Sharded | Memory Savings | Use Case |
|-------|----------------|----------------|----------|
| ZeRO-1 | Optimizer states | ~4x | Basic multi-GPU |
| ZeRO-2 | + Gradients | ~8x | **Recommended** |
| ZeRO-3 | + Parameters | ~16x | 70B+ models |

#### Multi-GPU with FSDP

```bash
accelerate launch \
  --config_file configs/gpu/fsdp/accelerate_fsdp.yaml \
  scripts/gpu/train_sft.py \
  --config configs/gpu/sft_config.yaml
```

**FSDP vs DeepSpeed:**
| Aspect | FSDP | DeepSpeed |
|--------|------|-----------|
| Setup | Easier | More complex |
| PyTorch Native | Yes | No |
| Best For | 7B-70B | 70B+ |
| HuggingFace Rec | Primary | Alternative |

---

### Step 2: GPU DPO Training (1 day) - Math Only

**Purpose:** Learn from preference pairs (correct vs incorrect solutions).

```bash
# Requires SFT checkpoint
python scripts/gpu/train_dpo.py \
  --config configs/gpu/dpo_config.yaml
```

**Config Explained:**
```yaml
model:
  name: "outputs/gpu/checkpoints/sft/final"  # Use SFT checkpoint

dpo:
  beta: 0.1               # KL divergence weight (0.1-0.5)
  loss_type: "sigmoid"    # sigmoid, hinge, ipo

training:
  learning_rate: 5.0e-7   # MUCH lower than SFT!
  num_train_epochs: 1     # DPO overfits quickly
```

**DPO Key Points:**
- Requires preference pairs (chosen/rejected)
- Reference model created automatically (2x memory)
- Lower learning rate than SFT
- Fewer epochs (1-2)

---

### Step 3: GPU GRPO Training (1-2 days) - Math Only

**Purpose:** Online RL using model's own generations.

```bash
# Requires DPO checkpoint
python scripts/gpu/train_grpo.py \
  --config configs/gpu/grpo_config.yaml
```

**Config Explained:**
```yaml
model:
  name: "outputs/gpu/checkpoints/dpo/final"  # Use DPO checkpoint

grpo:
  kl_coef: 0.05                # KL penalty (exploration vs exploitation)
  num_generations: 4           # Samples per prompt
  max_new_tokens: 512
  temperature: 0.7

training:
  learning_rate: 1.0e-6        # Very low for RL
```

**GRPO vs DPO vs PPO:**
| Method | Data | Reward Model | Memory | Stability |
|--------|------|--------------|--------|-----------|
| DPO | Offline (fixed pairs) | No | 2x | High |
| PPO | Online (generated) | Yes | 3-4x | Low |
| GRPO | Online (generated) | No | 2-3x | Medium-High |

---

## 9. Evaluation

### MLX Evaluation

```bash
# Translation (BLEU)
python scripts/mlx/evaluate.py \
  --task translation \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter outputs/mlx/adapters/korean_translation \
  --max-samples 100

# Math (Accuracy)
python scripts/mlx/evaluate.py \
  --task math \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter outputs/mlx/adapters/math_reasoning \
  --max-samples 100
```

### GPU Evaluation with lm-eval-harness

```bash
# GSM8K benchmark
lm_eval --model hf \
  --model_args pretrained=outputs/gpu/checkpoints/sft/final,dtype=bfloat16 \
  --tasks gsm8k \
  --batch_size 8 \
  --output_path outputs/gpu/logs/lm_eval

# Multiple benchmarks
lm_eval --model hf \
  --model_args pretrained=outputs/gpu/checkpoints/sft/final \
  --tasks gsm8k,math_algebra,minerva_math \
  --batch_size 8
```

### Translation Metrics Summary

| Metric | Tool | What It Measures |
|--------|------|------------------|
| BLEU | sacrebleu | N-gram overlap |
| chrF | sacrebleu | Character F-score |
| TER | sacrebleu | Edit distance |
| COMET | unbabel-comet | Semantic similarity |

### Math Metrics Summary

| Benchmark | Samples | Difficulty | Target |
|-----------|---------|------------|--------|
| GSM8K | 1,319 | Grade school | 85%+ |
| MATH | 5,000 | Competition | 30%+ |
| Minerva | Varies | Advanced | 20%+ |

---

## 10. Deployment

### Merge LoRA Adapters

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
)

# Load and merge adapters
model = PeftModel.from_pretrained(base_model, "outputs/gpu/checkpoints/sft/final")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("outputs/gpu/merged_models/final")
tokenizer.save_pretrained("outputs/gpu/merged_models/final")
```

### Deploy with vLLM

```bash
# Install vLLM
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
  --model outputs/gpu/merged_models/final \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096

# Test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "outputs/gpu/merged_models/final",
    "prompt": "Translate to English: 안녕하세요",
    "max_tokens": 100
  }'
```

### Deploy with SGLang

```bash
# Install SGLang
pip install sglang

# Start server (better for structured outputs)
python -m sglang.launch_server \
  --model-path outputs/gpu/merged_models/final \
  --port 8000

# Test with structured output
curl http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Solve: 2 + 2 = ",
    "sampling_params": {"max_new_tokens": 100}
  }'
```

### Deploy on Mac with MLX

```python
from mlx_lm import load, generate

# Load model with adapters
model, tokenizer = load(
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    adapter_path="outputs/mlx/adapters/korean_translation"
)

# Generate
response = generate(
    model, tokenizer,
    prompt="Translate to English: 안녕하세요",
    max_tokens=100
)
print(response)
```

### Upload to Hugging Face Hub

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="outputs/gpu/merged_models/final",
    repo_id="your-username/korean-english-translator",
    repo_type="model",
)
```

---

## 11. Complete Workflow Timeline

```
Week 1: Mac Development (MLX)
├── Day 1-2: Environment setup, data preparation
├── Day 3-4: Train translation model (SFT LoRA)
├── Day 5-6: Train math model (SFT LoRA)
└── Day 7: Evaluate, tune hyperparameters, export configs

Week 2: Cloud GPU Training
├── Day 1: Setup cloud instance, transfer data
├── Day 2-3: Translation SFT (full training)
├── Day 3-4: Math SFT (full training)
├── Day 4-5: Math DPO training
├── Day 5-6: Math GRPO training
└── Day 6-7: Final evaluation, merge adapters

Week 3: Deployment
├── Day 1-2: Merge models, upload to Hub
├── Day 3-4: Setup vLLM/SGLang serving
└── Day 5: Production testing, monitoring
```

### Cost Estimate

| Phase | Resource | Duration | Cost |
|-------|----------|----------|------|
| Mac Development | M2/M3 Mac | 7 days | $0 |
| GPU Training | A100 80GB | 50 hours | ~$55-100 |
| Inference Server | A10G | Ongoing | ~$1/hr |
| **Total Training** | | | **$55-100** |

---

## 12. Troubleshooting

### MLX Issues

**Out of Memory:**
```bash
# Reduce batch size
--batch-size 1

# Use smaller model
--model mlx-community/Phi-3.5-mini-instruct-4bit

# Enable gradient checkpointing (in config)
grad_checkpoint: true
```

**Model Not Found:**
```bash
# Check available MLX models
python -c "from huggingface_hub import list_models; print([m.id for m in list_models(filter='mlx-community')])"
```

### GPU Issues

**CUDA Out of Memory:**
```python
# Use QLoRA (4-bit)
quantization:
  enabled: true
  load_in_4bit: true

# Reduce batch size
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# Enable gradient checkpointing
gradient_checkpointing: true
```

**Flash Attention Not Working:**
```bash
# Check GPU compatibility (needs Ampere+)
python -c "import torch; print(torch.cuda.get_device_capability())"
# Needs (8, 0) or higher

# Fallback to eager attention
attn_implementation: "eager"
```

**DeepSpeed Issues:**
```bash
# Check installation
ds_report

# Common fix
pip uninstall deepspeed
pip install deepspeed --no-cache-dir
```

### Data Issues

**Dataset Loading Fails:**
```bash
# Clear cache
rm -rf ~/.cache/huggingface/datasets

# Re-download
python data/scripts/prepare_translation_data.py
```

**Tokenization Errors:**
```python
# Check tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
print(tok.encode("Test 안녕하세요"))
```

### Monitoring Issues

**Wandb Not Logging:**
```bash
# Login
wandb login

# Check API key
echo $WANDB_API_KEY

# Test connection
python -c "import wandb; wandb.init(project='test'); wandb.finish()"
```

---

## Quick Reference Commands

```bash
# === SETUP ===
./setup_project.sh                    # Create folders
source setup_env.sh mlx               # MLX environment
source setup_env.sh gpu               # GPU environment

# === DATA ===
python data/scripts/prepare_translation_data.py
python data/scripts/prepare_math_data.py

# === MLX TRAINING ===
python scripts/mlx/train_sft.py -c configs/mlx/sft_korean_translation.yaml
python scripts/mlx/train_sft.py -c configs/mlx/sft_math_reasoning.yaml

# === MLX EVALUATION ===
python scripts/mlx/evaluate.py -t translation -a outputs/mlx/adapters/korean_translation
python scripts/mlx/evaluate.py -t math -a outputs/mlx/adapters/math_reasoning

# === GPU TRAINING ===
python scripts/gpu/train_sft.py -c configs/gpu/sft_config.yaml
python scripts/gpu/train_dpo.py -c configs/gpu/dpo_config.yaml
python scripts/gpu/train_grpo.py -c configs/gpu/grpo_config.yaml

# === MULTI-GPU ===
accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json scripts/gpu/train_sft.py -c configs/gpu/sft_config.yaml
accelerate launch --config_file configs/gpu/fsdp/accelerate_fsdp.yaml scripts/gpu/train_sft.py -c configs/gpu/sft_config.yaml

# === EVALUATION ===
lm_eval --model hf --model_args pretrained=outputs/gpu/checkpoints/sft/final --tasks gsm8k

# === DEPLOYMENT ===
python -m vllm.entrypoints.openai.api_server --model outputs/gpu/merged_models/final --port 8000
```

---

*Last updated: December 2024*
