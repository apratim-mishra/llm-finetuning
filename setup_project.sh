#!/bin/bash
# =============================================================================
# Project Setup Script - Creates folder structure for MLX → GPU training pipeline
# Run: chmod +x setup_project.sh && ./setup_project.sh
# =============================================================================

set -e

echo "🚀 Setting up LLM Fine-tuning Project Structure..."

# Create main directories
mkdir -p configs/{mlx,gpu/{deepspeed,fsdp},evaluation}
mkdir -p data/{raw/{korean_english,math},processed/{korean_english,math},scripts}
mkdir -p src/{data,training,evaluation,rewards,inference}
mkdir -p scripts/{mlx,gpu,common}
mkdir -p notebooks
mkdir -p outputs/{mlx/{checkpoints,adapters,logs},gpu/{checkpoints,merged_models,logs}}
mkdir -p tests
mkdir -p requirements

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/data/__init__.py
touch src/training/__init__.py
touch src/evaluation/__init__.py
touch src/rewards/__init__.py
touch src/inference/__init__.py

# Create placeholder files
touch data/raw/korean_english/.gitkeep
touch data/raw/math/.gitkeep
touch data/processed/korean_english/.gitkeep
touch data/processed/math/.gitkeep
touch outputs/mlx/checkpoints/.gitkeep
touch outputs/mlx/adapters/.gitkeep
touch outputs/mlx/logs/.gitkeep
touch outputs/gpu/checkpoints/.gitkeep
touch outputs/gpu/merged_models/.gitkeep
touch outputs/gpu/logs/.gitkeep

echo "✅ Folder structure created!"
echo ""
echo "📁 Project Structure:"
find . -type d | head -40 | sed 's/^/  /'

echo ""
echo "Next steps:"
echo "  1. Run: source setup_env.sh mlx    (for Mac MLX environment)"
echo "  2. Run: python data/scripts/prepare_translation_data.py"
echo "  3. Run: python data/scripts/prepare_math_data.py"
