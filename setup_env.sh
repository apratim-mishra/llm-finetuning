#!/bin/bash
# =============================================================================
# Environment Setup Script - Supports both uv (faster) and venv (traditional)
# Usage: source setup_env.sh [mlx|gpu] [--use-venv]
# =============================================================================

ENV_TYPE=${1:-mlx}
USE_VENV=${2:-""}

# Allow overriding Python version/binary; default to repo's .python-version (3.11).
PYTHON_VERSION=${LLM_FT_PYTHON_VERSION:-$(cat .python-version 2>/dev/null || echo "3.11")}
PYTHON_BIN=${LLM_FT_PYTHON_BIN:-python${PYTHON_VERSION}}
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    PYTHON_BIN=${LLM_FT_PYTHON_BIN:-python3}
fi

# Detect package manager
if command -v uv &> /dev/null && [ "$USE_VENV" != "--use-venv" ]; then
    PKG_MANAGER="uv"
    echo "Using uv package manager (faster)"
else
    PKG_MANAGER="venv"
    echo "Using traditional venv"
fi

# Set environment name based on type
if [ "$ENV_TYPE" = "mlx" ]; then
    ENV_NAME=".venv-mlx"
    REQ_FILE="requirements/requirements-mlx.txt"
    echo "Setting up Mac MLX environment..."
elif [ "$ENV_TYPE" = "gpu" ]; then
    ENV_NAME=".venv-gpu"
    REQ_FILE="requirements/requirements-gpu.txt"
    echo "Setting up Cloud GPU environment..."
else
    echo "Invalid environment type. Use: mlx or gpu"
    return 1
fi

# Create and activate environment
if [ "$PKG_MANAGER" = "uv" ]; then
    # Using uv
    if [ ! -d "$ENV_NAME" ]; then
        echo "Creating virtual environment with uv..."
        uv venv "$ENV_NAME" --python "$PYTHON_VERSION"
    fi
    
    echo "Activating $ENV_NAME..."
    source "$ENV_NAME/bin/activate"
    
    echo "Installing dependencies with uv..."
    uv pip install -r "$REQ_FILE"
    
else
    # Using traditional venv
    if [ ! -d "$ENV_NAME" ]; then
        echo "Creating virtual environment..."
        "$PYTHON_BIN" -m venv "$ENV_NAME"
    fi
    
    echo "Activating $ENV_NAME..."
    source "$ENV_NAME/bin/activate"
    
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    echo "Installing dependencies..."
    pip install -r "$REQ_FILE"
fi

# Verify installation
echo ""
echo "Environment setup complete!"
echo "   Environment: $ENV_NAME"
echo "   Python: $(python --version)"
echo ""

if [ "$ENV_TYPE" = "mlx" ]; then
    echo "Verifying MLX installation..."
    python -c "import mlx; print(f'   MLX version: {mlx.__version__}')" 2>/dev/null || echo "   Warning: MLX not installed (expected on non-Mac)"
    python -c "import mlx_lm; print('   mlx-lm available')" 2>/dev/null || echo "   Warning: mlx-lm not installed"
else
    echo "Verifying GPU packages..."
    python -c "import torch; print(f'   PyTorch: {torch.__version__}')" 2>/dev/null || echo "   Warning: PyTorch not installed"
    python -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || true
fi

echo ""
echo "Next steps:"
if [ "$ENV_TYPE" = "mlx" ]; then
    echo "   1. Prepare data: python data/scripts/prepare_translation_data.py"
    echo "   2. Train: python scripts/mlx/train_sft.py --config configs/mlx/sft_korean_translation.yaml"
else
    echo "   1. Train SFT: python scripts/gpu/train_sft.py --config configs/gpu/sft_config.yaml"
    echo "   2. Train DPO: python scripts/gpu/train_dpo.py --config configs/gpu/dpo_config.yaml"
fi
