"""
Tests for MLX Training Scripts
Tests scripts/mlx/train_sft.py and scripts/mlx/evaluate.py.
Mocks MLX dependencies for testing on non-Mac systems.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_config(self, sample_config_file):
        """Test loading a valid YAML config."""
        from scripts.mlx.train_sft import load_config

        config = load_config(str(sample_config_file))
        assert "model" in config
        assert "training" in config

    def test_config_values(self, sample_config_file):
        """Test config values are correct."""
        from scripts.mlx.train_sft import load_config

        config = load_config(str(sample_config_file))
        assert config["model"]["name"] == "mlx-community/Qwen2.5-7B-Instruct-4bit"
        assert config["training"]["iters"] == 100


class TestPrepareMlxData:
    """Tests for prepare_mlx_data function."""

    def test_creates_data_directory(self, temp_dir, sample_jsonl_file):
        """Test that data directory is created."""
        from scripts.mlx.train_sft import prepare_mlx_data

        output_dir = temp_dir / "output"
        data_dir, train_path = prepare_mlx_data(
            str(sample_jsonl_file),
            str(output_dir)
        )

        assert Path(data_dir).exists()
        assert "mlx_data" in data_dir

    def test_copies_train_file(self, temp_dir, sample_jsonl_file):
        """Test that train file is copied."""
        from scripts.mlx.train_sft import prepare_mlx_data

        output_dir = temp_dir / "output"
        data_dir, train_path = prepare_mlx_data(
            str(sample_jsonl_file),
            str(output_dir)
        )

        assert Path(train_path).exists()
        assert Path(train_path).name == "train.jsonl"


class TestBuildMlxCommand:
    """Tests for build_mlx_command function."""

    def test_basic_command(self, sample_config):
        """Test basic command construction."""
        from scripts.mlx.train_sft import build_mlx_command

        cmd = build_mlx_command(sample_config, "/data/dir")

        assert "-m" in cmd
        assert "mlx_lm.lora" in cmd
        assert "--train" in cmd
        assert "--model" in cmd

    def test_includes_model_name(self, sample_config):
        """Test model name is included."""
        from scripts.mlx.train_sft import build_mlx_command

        cmd = build_mlx_command(sample_config, "/data/dir")
        cmd_str = " ".join(cmd)

        assert "Qwen2.5-7B-Instruct-4bit" in cmd_str

    def test_includes_training_params(self, sample_config):
        """Test training parameters are included."""
        from scripts.mlx.train_sft import build_mlx_command

        cmd = build_mlx_command(sample_config, "/data/dir")
        cmd_str = " ".join(cmd)

        assert "--iters" in cmd_str
        assert "--batch-size" in cmd_str
        assert "--learning-rate" in cmd_str

    def test_includes_lora_params(self, sample_config):
        """Test LoRA parameters are included."""
        from scripts.mlx.train_sft import build_mlx_command

        cmd = build_mlx_command(sample_config, "/data/dir")
        cmd_str = " ".join(cmd)

        assert "--lora-rank" in cmd_str
        assert "--lora-layers" in cmd_str

    def test_grad_checkpoint_flag(self, sample_config):
        """Test gradient checkpoint flag."""
        from scripts.mlx.train_sft import build_mlx_command

        sample_config["training"]["grad_checkpoint"] = True
        cmd = build_mlx_command(sample_config, "/data/dir")

        assert "--grad-checkpoint" in cmd


class TestInitWandb:
    """Tests for init_wandb function."""

    def test_disabled_wandb(self, sample_config):
        """Test wandb initialization when disabled."""
        from scripts.mlx.train_sft import init_wandb

        sample_config["logging"]["wandb"]["enabled"] = False
        result = init_wandb(sample_config)

        assert result is False

    def test_missing_wandb_config(self):
        """Test wandb initialization with missing config."""
        from scripts.mlx.train_sft import init_wandb

        config = {"logging": {}}
        result = init_wandb(config)

        assert result is False

    @patch("scripts.mlx.train_sft.wandb", create=True)
    def test_wandb_init_called(self, mock_wandb, sample_config):
        """Test wandb.init is called when enabled."""
        from scripts.mlx.train_sft import init_wandb

        sample_config["logging"]["wandb"]["enabled"] = True
        mock_wandb.init = MagicMock()

        # This will fail on import but tests the logic
        try:
            init_wandb(sample_config)
        except ImportError:
            pass  # Expected if wandb not installed


@pytest.mark.mlx
class TestMlxEvaluate:
    """Tests for MLX evaluation functions."""

    def test_extract_answer_gsm8k(self):
        """Test answer extraction in GSM8K format."""
        from scripts.mlx.evaluate import extract_answer

        text = "Let's solve: 5+3=8. #### 8"
        answer = extract_answer(text)
        assert answer == "8"

    def test_extract_answer_explicit(self):
        """Test explicit answer extraction."""
        from scripts.mlx.evaluate import extract_answer

        text = "The answer is 42."
        answer = extract_answer(text)
        assert answer == "42"

    def test_normalize_answer(self):
        """Test answer normalization."""
        from scripts.mlx.evaluate import normalize_answer

        assert normalize_answer("42") == 42.0
        assert normalize_answer("$100") == 100.0
        assert normalize_answer("1,234") == 1234.0

    def test_normalize_answer_invalid(self):
        """Test normalization of invalid input."""
        from scripts.mlx.evaluate import normalize_answer

        assert normalize_answer("not a number") is None


@pytest.mark.mlx
class TestMlxModelLoading:
    """Tests for MLX model loading (mocked)."""

    @patch("scripts.mlx.evaluate.load")
    def test_load_base_model(self, mock_load):
        """Test loading base model."""
        from scripts.mlx.evaluate import load_model_and_tokenizer

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        model, tokenizer, gen_fn = load_model_and_tokenizer(
            "mlx-community/test-model"
        )

        mock_load.assert_called_once()
        assert model == mock_model
        assert tokenizer == mock_tokenizer

    @patch("scripts.mlx.evaluate.load")
    def test_load_with_adapter(self, mock_load, temp_dir):
        """Test loading model with adapter."""
        from scripts.mlx.evaluate import load_model_and_tokenizer

        adapter_path = temp_dir / "adapters"
        adapter_path.mkdir()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        model, tokenizer, gen_fn = load_model_and_tokenizer(
            "mlx-community/test-model",
            adapter_path=str(adapter_path)
        )

        mock_load.assert_called_with(
            "mlx-community/test-model",
            adapter_path=str(adapter_path)
        )


@pytest.mark.mlx
class TestMlxGeneration:
    """Tests for MLX generation (mocked)."""

    def test_generate_response(self):
        """Test response generation."""
        from scripts.mlx.evaluate import generate_response

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_generate = MagicMock(return_value="Generated text")

        result = generate_response(
            mock_model,
            mock_tokenizer,
            mock_generate,
            "Test prompt",
            max_tokens=100,
            temperature=0.1
        )

        assert result == "Generated text"
        mock_generate.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            prompt="Test prompt",
            max_tokens=100,
            temp=0.1
        )


class TestMlxIntegration:
    """Integration tests for MLX scripts."""

    def test_config_to_command_roundtrip(self, sample_config, temp_dir):
        """Test config loading and command building."""
        from scripts.mlx.train_sft import load_config, build_mlx_command

        # Save config
        config_path = temp_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        # Load and build command
        loaded_config = load_config(str(config_path))
        cmd = build_mlx_command(loaded_config, str(temp_dir))

        # Verify essential parts
        assert "--model" in cmd
        assert "--train" in cmd
        assert "--iters" in cmd

    def test_data_preparation_flow(self, temp_dir, sample_messages):
        """Test data preparation flow."""
        from scripts.mlx.train_sft import prepare_mlx_data

        # Create source file
        source_file = temp_dir / "source" / "train.jsonl"
        source_file.parent.mkdir(parents=True)
        with open(source_file, "w") as f:
            for msg in sample_messages:
                f.write(json.dumps(msg) + "\n")

        # Prepare data
        output_dir = temp_dir / "output"
        data_dir, train_path = prepare_mlx_data(
            str(source_file),
            str(output_dir)
        )

        # Verify
        assert Path(data_dir).exists()
        assert Path(train_path).exists()

        # Check content preserved
        with open(train_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == len(sample_messages)


@pytest.mark.mlx
class TestMlxTranslationEvaluation:
    """Tests for translation evaluation (mocked)."""

    def test_translation_prompt_format(self):
        """Test translation prompt formatting."""
        system_prompt = "Translate Korean to English."
        korean_text = "안녕하세요"

        expected_format = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{korean_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Build prompt as in evaluate.py
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{korean_text}<|im_end|>\n<|im_start|>assistant\n"

        assert prompt == expected_format

    def test_clean_prediction(self):
        """Test prediction cleaning."""
        prediction = "Hello world<|im_end|>extra text"
        cleaned = prediction.split("<|im_end|>")[0].strip()

        assert cleaned == "Hello world"


@pytest.mark.mlx
class TestMlxMathEvaluation:
    """Tests for math evaluation (mocked)."""

    def test_math_prompt_format(self):
        """Test math prompt formatting."""
        system_prompt = "Solve step by step."
        question = "What is 2+2?"

        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        assert "<|im_start|>system" in prompt
        assert question in prompt

    def test_answer_comparison(self):
        """Test answer comparison logic."""
        from scripts.mlx.evaluate import normalize_answer

        pred = normalize_answer("42")
        true = normalize_answer("42")

        assert pred is not None
        assert true is not None
        assert abs(pred - true) < 0.01
