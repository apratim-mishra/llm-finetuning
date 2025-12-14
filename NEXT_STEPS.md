# Next Steps Roadmap (Remaining / Optional)

The original roadmap items in this repo have been implemented (serving presets/docs, LoRA merge, ONNX wrapper,
GPU profiles + config merging, dataset manifests, preference-pair generation, task-level eval entrypoints,
and CPU-only smoke tests).

If you want to keep expanding beyond the current end-to-end pipeline, here are the highest-impact additions.

## Recently Completed

### Model Options (Qwen3, Gemma3, Llama3.3)

- Added newer model alternatives to MLX configs (`configs/mlx/*.yaml`)
- Added newer model alternatives to GPU configs with VRAM requirements
- See README.md "Supported Models" section for full list

### Preference Optimization Methods

- Added SimPO trainer (`scripts/gpu/train_simpo.py`) - reference-free, stable with noisy data
- Added ORPO trainer (`scripts/gpu/train_orpo.py`) - combines SFT + preference in one step
- Added KTO trainer (`scripts/gpu/train_kto.py`) - works with unpaired data, asymmetric loss
- Created corresponding configs in `configs/gpu/`

### Evaluation Enhancements

- Added MATH, MathQA, ASDiv, LogiQA benchmarks to `evaluate_lm_harness.py`
- Added BLEURT and xCOMET metrics to `src/evaluation/translation_metrics.py`

### Preference Generation Improvements

- Added full provenance metadata to `generate_math_preferences.py` output
- Added step-level reward function (PRM-style) to `src/rewards/math_reward.py`
- vLLM/SGLang already use efficient `n` sampling (k candidates in one call)

---

## 1) CI (recommended)

- Add a GitHub Actions workflow that runs `make lint` + `make test` on PRs (CPU-only).
- Keep GPU/MLX installs optional so CI stays fast and reliable.

## 2) One-command pipelines

- Add a thin "pipeline runner" that chains `data -> train -> eval -> merge -> serve` per task.
- Resolve stage dependencies automatically (e.g., GRPO uses the DPO adapter, DPO uses the SFT adapter).

## 3) VRAM/config preflight (quality-of-life)

- Add a VRAM estimator + warnings before training/serving.
- Auto-suggest a `configs/gpu/profiles/*` preset based on `nvidia-smi` and chosen `max_model_len`.

## 4) TensorRT-LLM automation (advanced)

`scripts/export/export_tensorrt_llm.md` is a runbook. Next step is a reproducible automation script that:

- pins TensorRT-LLM versions and build steps
- writes artifacts under `outputs/gpu/tensorrt_llm/*`
- provides a small "wiring-only" smoke test (no build in CI)

## 5) Advanced RLHF/RLVR (future)

Consider integrating more advanced RL methods as they mature:

- **GRPO variants**: Scaf-GRPO (scaffolded hints), NGRPO (negative-enhanced), CPPO (completion pruning)
- **Process Reward Models**: Full PRM training pipeline with step-level labels
- **OpenRLHF integration**: For production-scale RLHF with Ray + vLLM
- **Online preference generation**: Generate + score during training (not just offline)

## 6) Multi-task / Multi-lingual Expansion

- Add more language pairs for translation task
- Add code generation task (HumanEval, MBPP evaluation)
- Support for mixture-of-experts models (Qwen3-Next MoE, etc.)
