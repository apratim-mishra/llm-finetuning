# Repository Guidelines

Use context7 if needed when I need code generation, setup or configuration steps, or
library/API documentation.

## Project Structure & Module Organization

The repo centers on `src/` for reusable data, evaluation, and reward modules, while runnable entry points live under `scripts/mlx/` and `scripts/gpu/` for Apple Silicon and CUDA workflows. Training and evaluation knobs are captured in YAML inside `configs/` (`mlx`, `gpu`, `deepspeed`, `fsdp`, `evaluation`). Data assets are staged in `data/raw/`, `data/processed/`, and `data/scripts/`, with generated checkpoints in `outputs/`. Tests sit in `tests/` and mirror the directory names they validate (e.g., `test_data_loader.py`).

## Build, Test, and Development Commands

Use `make setup-project` once, then `source setup_env.sh mlx|gpu` to activate the right toolchain. Data prep runs through `make data-translation` and `make data-math`. Local fine-tuning relies on `make train-mlx-translation` or `python scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml`. Run reinforcement stages with the matching `train_dpo.py` or `train_grpo.py` configs. `make lint` executes Black and Ruff, while `make test` runs Pytest with coverage.

## Coding Style & Naming Conventions

Target Python 3.10+, follow Black formatting (4-space indent, 100-char lines), and let Ruff enforce Pycodestyle, Pyflakes, Bugbear, and pep8-naming; fix imports with isort’s Black profile. Modules and files stay snake_case, classes use PascalCase, and configs adopt kebabed YAML keys already present in `configs/`. Prefer explicit dataclasses for structured outputs (see `src/rewards/math_reward.py`) and keep CLI args descriptive (e.g., `--backend`, `--tensor-parallel`).

## Testing Guidelines

Pytest discovers files named `test_*.py` and functions `test_*`, so mirror the unit under test and colocate fixtures in `tests/conftest.py`. `make test` (or `pytest tests/ -v`) enforces `--cov=src` and emits HTML coverage to `outputs/coverage`; avoid regressing coverage without justification. When adding scripts, provide lightweight mocks or sample JSONL fixtures under `tests/` or `data/tests/` so CI can run without GPUs.

## Commit & Pull Request Guidelines

Follow the existing history by keeping commit titles short, imperative, and scoped (e.g., `data: add translation sampling helper`). Reference issue IDs in the body, summarize config or hardware impacts, and mention whether commands such as `make test` or `train-gpu-sft` were exercised. PRs should link to datasets used, attach Wandb/MLflow run URLs if available, and highlight any new configs or artifacts created under `outputs/` so reviewers can reproduce results.

## Security & Configuration Tips

Never commit `.env` contents; copy `.env.example`, populate HF, Wandb, and cloud credentials locally, and rely on environment variables inside `setup_env.sh`. Inspect generated files in `outputs/` before pushing to avoid leaking adapters or checkpoints. When sharing configs, strip API tokens and instead reference parameter names documented in README’s credentials section.
