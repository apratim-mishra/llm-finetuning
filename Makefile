# =============================================================================
# Makefile - Common commands for LLM Fine-tuning Pipeline
# =============================================================================

.PHONY: setup-mlx setup-gpu data-translation data-math data-math-preferences train-mlx-translation train-mlx-math \
        train-gpu-sft-math \
        eval-mlx eval-translation eval-math train-gpu-sft train-gpu-dpo train-gpu-grpo export-merge-translation export-merge-math export-onnx clean help

# Default Python
PYTHON := python

# Make runs each recipe line in its own shell; use bash for `source`.
SHELL := /bin/bash

# =============================================================================
# Environment Setup
# =============================================================================

setup-project:
	@echo "Setting up project structure..."
	chmod +x setup_project.sh
	./setup_project.sh

setup-mlx:
	@echo "Setting up Mac MLX environment..."
	bash -c "source ./setup_env.sh mlx"

setup-gpu:
	@echo "Setting up GPU environment..."
	bash -c "source ./setup_env.sh gpu"

# =============================================================================
# Data Preparation
# =============================================================================

data-translation:
	@echo "Preparing Korean-English translation data..."
	$(PYTHON) data/scripts/prepare_translation_data.py

data-translation-full:
	@echo "Preparing full Korean-English translation data..."
	$(PYTHON) data/scripts/prepare_translation_data.py --full

data-math:
	@echo "Preparing math reasoning data..."
	$(PYTHON) data/scripts/prepare_math_data.py

data-math-preferences:
	@echo "Generating math preference pairs (DPO)..."
	$(PYTHON) data/scripts/generate_math_preferences.py \
		--input data/processed/math/grpo_prompts.jsonl \
		--output-dir data/processed/math \
		--backend vllm \
		--model Qwen/Qwen2.5-7B-Instruct

data-math-full:
	@echo "Preparing full math reasoning data..."
	$(PYTHON) data/scripts/prepare_math_data.py --full

data-all: data-translation data-math
	@echo "All data prepared!"

# =============================================================================
# MLX Training (Mac)
# =============================================================================

train-mlx-translation:
	@echo "Training Korean-English translation model (MLX)..."
	$(PYTHON) scripts/mlx/train_sft.py --config configs/mlx/sft_korean_translation.yaml

train-mlx-math:
	@echo "Training math reasoning model (MLX)..."
	$(PYTHON) scripts/mlx/train_sft.py --config configs/mlx/sft_math_reasoning.yaml

# =============================================================================
# MLX Evaluation (Mac)
# =============================================================================

eval-mlx-translation:
	@echo "Evaluating translation model (MLX)..."
	$(PYTHON) scripts/mlx/evaluate.py --task translation \
		--adapter outputs/mlx/adapters/korean_translation

eval-mlx-math:
	@echo "Evaluating math model (MLX)..."
	$(PYTHON) scripts/mlx/evaluate.py --task math \
		--adapter outputs/mlx/adapters/math_reasoning

# =============================================================================
# Evaluation (Task-level)
# =============================================================================

eval-translation:
	@echo "Evaluating translation task (GPU inference + metrics)..."
	$(PYTHON) scripts/eval/eval_translation.py --config configs/evaluation/translation.yaml

eval-math:
	@echo "Evaluating math task (GPU inference + reward metrics)..."
	$(PYTHON) scripts/eval/eval_math.py --config configs/evaluation/math.yaml

# =============================================================================
# GPU Training (Cloud)
# =============================================================================

train-gpu-sft:
	@echo "Training SFT model (GPU, translation)..."
	$(PYTHON) scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml

train-gpu-sft-math:
	@echo "Training SFT model (GPU, math)..."
	$(PYTHON) scripts/gpu/train_sft.py --config configs/gpu/sft_math_config.yaml

train-gpu-sft-deepspeed:
	@echo "Training SFT model with DeepSpeed (GPU)..."
	accelerate launch --config_file configs/gpu/deepspeed/ds_zero2.json \
		scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml

train-gpu-sft-fsdp:
	@echo "Training SFT model with FSDP (GPU)..."
	accelerate launch --config_file configs/gpu/fsdp/accelerate_fsdp.yaml \
		scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml

train-gpu-dpo:
	@echo "Training DPO model (GPU)..."
	$(PYTHON) scripts/gpu/train_dpo.py --config configs/gpu/dpo_config.yaml

train-gpu-grpo:
	@echo "Training GRPO model (GPU)..."
	$(PYTHON) scripts/gpu/train_grpo.py --config configs/gpu/grpo_config.yaml

# Full math pipeline
train-math-pipeline: train-gpu-sft-math train-gpu-dpo train-gpu-grpo
	@echo "Math training pipeline complete!"

# =============================================================================
# Export / Packaging
# =============================================================================

export-merge-translation:
	@echo "Merging translation LoRA adapter into base model..."
	$(PYTHON) scripts/gpu/merge_lora.py \
		--adapter outputs/gpu/checkpoints/sft/final \
		--output outputs/gpu/merged_models/translation_sft_merged

export-merge-math:
	@echo "Merging math GRPO LoRA adapter into base model..."
	$(PYTHON) scripts/gpu/merge_lora.py \
		--adapter outputs/gpu/checkpoints/math_grpo/final \
		--output outputs/gpu/merged_models/math_grpo_merged

export-onnx:
	@echo "Exporting merged model to ONNX (best-effort)..."
	$(PYTHON) scripts/export/export_onnx.py --config configs/export/onnx.yaml

# =============================================================================
# Evaluation (GPU)
# =============================================================================

eval-gpu-lm-eval:
	@echo "Running lm-evaluation-harness..."
	lm_eval --model hf \
		--model_args pretrained=outputs/gpu/checkpoints/sft/final,dtype=bfloat16 \
		--tasks gsm8k \
		--batch_size 8 \
		--output_path outputs/gpu/logs/lm_eval

# =============================================================================
# Utilities
# =============================================================================

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

clean-outputs:
	@echo "Cleaning outputs..."
	rm -rf outputs/mlx/checkpoints/*
	rm -rf outputs/mlx/adapters/*
	rm -rf outputs/gpu/checkpoints/*

lint:
	@echo "Running linters..."
	black src/ scripts/ data/scripts/
	ruff check src/ scripts/ data/scripts/

test:
	@echo "Running tests..."
	pytest tests/ -v

# =============================================================================
# Help
# =============================================================================

help:
	@echo "LLM Fine-tuning Pipeline Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup-project     - Create folder structure"
	@echo "  make setup-mlx         - Setup Mac MLX environment"
	@echo "  make setup-gpu         - Setup GPU environment"
	@echo ""
	@echo "Data Preparation:"
	@echo "  make data-translation  - Prepare Korean-English data (subset)"
	@echo "  make data-math         - Prepare math data (subset)"
	@echo "  make data-math-preferences - Generate DPO preference pairs (GPU)"
	@echo "  make data-all          - Prepare all data"
	@echo ""
	@echo "MLX Training (Mac):"
	@echo "  make train-mlx-translation  - Train translation model"
	@echo "  make train-mlx-math         - Train math model (SFT only)"
	@echo ""
	@echo "MLX Evaluation (Mac):"
	@echo "  make eval-mlx-translation   - Evaluate translation (BLEU)"
	@echo "  make eval-mlx-math          - Evaluate math (accuracy)"
	@echo ""
	@echo "Task-level Evaluation:"
	@echo "  make eval-translation       - GPU inference + translation metrics"
	@echo "  make eval-math              - GPU inference + math reward metrics"
	@echo ""
	@echo "GPU Training (Cloud):"
	@echo "  make train-gpu-sft          - Train SFT (translation)"
	@echo "  make train-gpu-sft-math     - Train SFT (math)"
	@echo "  make train-gpu-sft-deepspeed- Train SFT with DeepSpeed"
	@echo "  make train-gpu-sft-fsdp     - Train SFT with FSDP"
	@echo "  make train-gpu-dpo          - Train DPO"
	@echo "  make train-gpu-grpo         - Train GRPO"
	@echo ""
	@echo "Export / Packaging:"
	@echo "  make export-merge-translation - Merge translation LoRA into base"
	@echo "  make export-merge-math        - Merge math GRPO LoRA into base"
	@echo "  make export-onnx              - Export merged model to ONNX"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean             - Clean temporary files"
	@echo "  make lint              - Run linters"
	@echo "  make test              - Run tests"
