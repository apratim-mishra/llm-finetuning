"""
Pytest Configuration and Fixtures
Shared fixtures for all tests.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_messages() -> list[dict]:
    """Sample ChatML-format messages for testing."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ]
        },
    ]


@pytest.fixture
def sample_translation_messages() -> list[dict]:
    """Sample Korean-English translation messages."""
    return [
        {
            "messages": [
                {"role": "system", "content": "Translate Korean to English."},
                {"role": "user", "content": "안녕하세요"},
                {"role": "assistant", "content": "Hello"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "Translate Korean to English."},
                {"role": "user", "content": "감사합니다"},
                {"role": "assistant", "content": "Thank you"},
            ]
        },
    ]


@pytest.fixture
def sample_math_problems() -> list[dict]:
    """Sample math problems in GSM8K format."""
    return [
        {
            "question": "John has 5 apples. He buys 3 more. How many apples does he have?",
            "answer": "8",
        },
        {
            "question": "A store has 100 items. They sell 30. How many remain?",
            "answer": "70",
        },
    ]


@pytest.fixture
def sample_preference_pairs() -> list[dict]:
    """Sample preference pairs for DPO training."""
    return [
        {
            "prompt": "Explain Python lists.",
            "chosen": "Python lists are ordered, mutable collections that can hold items of different types.",
            "rejected": "Lists are things in Python.",
        },
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of AI where systems learn patterns from data.",
            "rejected": "ML is computers being smart.",
        },
    ]


@pytest.fixture
def sample_jsonl_file(temp_dir: Path, sample_messages: list[dict]) -> Path:
    """Create a sample JSONL file."""
    filepath = temp_dir / "sample.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in sample_messages:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return filepath


@pytest.fixture
def sample_train_val_files(temp_dir: Path, sample_messages: list[dict]) -> tuple[Path, Path]:
    """Create sample train and validation JSONL files."""
    train_path = temp_dir / "train.jsonl"
    val_path = temp_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for item in sample_messages:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_messages[0], ensure_ascii=False) + "\n")

    return train_path, val_path


@pytest.fixture
def sample_math_jsonl(temp_dir: Path, sample_math_problems: list[dict]) -> Path:
    """Create a sample math problems JSONL file."""
    filepath = temp_dir / "math_test.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in sample_math_problems:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return filepath


@pytest.fixture
def sample_config() -> dict:
    """Sample MLX training configuration."""
    return {
        "model": {
            "name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "max_seq_length": 2048,
        },
        "lora": {
            "rank": 32,
            "layers": 16,
        },
        "training": {
            "iters": 100,
            "batch_size": 2,
            "learning_rate": 2e-5,
            "grad_checkpoint": False,
        },
        "checkpoint": {
            "output_dir": "outputs/mlx/adapters",
            "save_every": 50,
        },
        "data": {
            "train_file": "data/processed/train.jsonl",
        },
        "validation": {
            "val_batches": 10,
        },
        "logging": {
            "wandb": {
                "enabled": False,
            }
        },
    }


@pytest.fixture
def sample_config_file(temp_dir: Path, sample_config: dict) -> Path:
    """Create a sample config YAML file."""
    import yaml

    filepath = temp_dir / "config.yaml"
    with open(filepath, "w") as f:
        yaml.dump(sample_config, f)
    return filepath


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "mlx: mark test as requiring MLX framework")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Skip MLX tests if not on Apple Silicon."""
    import platform

    is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"

    skip_mlx = pytest.mark.skip(reason="MLX tests require Apple Silicon Mac")

    for item in items:
        if "mlx" in item.keywords and not is_mac_arm:
            item.add_marker(skip_mlx)
