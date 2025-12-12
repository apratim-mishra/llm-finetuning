# Next Steps Roadmap (Remaining / Optional)

The original roadmap items in this repo have been implemented (serving presets/docs, LoRA merge, ONNX wrapper,
GPU profiles + config merging, dataset manifests, preference-pair generation, task-level eval entrypoints,
and CPU-only smoke tests).

If you want to keep expanding beyond the current end-to-end pipeline, here are the highest-impact additions.

## 1) CI (recommended)

- Add a GitHub Actions workflow that runs `make lint` + `make test` on PRs (CPU-only).
- Keep GPU/MLX installs optional so CI stays fast and reliable.

## 2) One-command pipelines

- Add a thin “pipeline runner” that chains `data → train → eval → merge → serve` per task.
- Resolve stage dependencies automatically (e.g., GRPO uses the DPO adapter, DPO uses the SFT adapter).

## 3) Preference generation from model sampling (better DPO)

`data/scripts/generate_math_preferences.py` supports offline `candidates` today. Next, add an option to:
- generate `k` candidates per prompt via vLLM/SGLang (batch)
- score with `src/rewards/math_reward.py`
- write provenance (model id, sampling params, reward function) into the output

## 4) VRAM/config preflight (quality-of-life)

- Add a VRAM estimator + warnings before training/serving.
- Auto-suggest a `configs/gpu/profiles/*` preset based on `nvidia-smi` and chosen `max_model_len`.

## 5) TensorRT-LLM automation (advanced)

`scripts/export/export_tensorrt_llm.md` is a runbook. Next step is a reproducible automation script that:
- pins TensorRT-LLM versions and build steps
- writes artifacts under `outputs/gpu/tensorrt_llm/*`
- provides a small “wiring-only” smoke test (no build in CI)
