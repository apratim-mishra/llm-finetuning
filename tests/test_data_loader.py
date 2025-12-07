"""
Tests for Data Loading Utilities
Tests src/data/data_loader.py functions.
"""

import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset


class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_load_valid_jsonl(self, sample_jsonl_file):
        """Test loading a valid JSONL file."""
        from src.data.data_loader import load_jsonl

        data = load_jsonl(sample_jsonl_file)
        assert len(data) == 2
        assert "messages" in data[0]

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading a file that doesn't exist."""
        from src.data.data_loader import load_jsonl

        with pytest.raises(FileNotFoundError):
            load_jsonl(temp_dir / "nonexistent.jsonl")

    def test_load_empty_file(self, temp_dir):
        """Test loading an empty file."""
        from src.data.data_loader import load_jsonl

        filepath = temp_dir / "empty.jsonl"
        filepath.write_text("")

        data = load_jsonl(filepath)
        assert data == []

    def test_load_with_malformed_json(self, temp_dir):
        """Test handling of malformed JSON lines."""
        from src.data.data_loader import load_jsonl

        filepath = temp_dir / "malformed.jsonl"
        with open(filepath, "w") as f:
            f.write('{"valid": "json"}\n')
            f.write("this is not valid json\n")
            f.write('{"another": "valid"}\n')

        data = load_jsonl(filepath)
        assert len(data) == 2


class TestSaveJsonl:
    """Tests for save_jsonl function."""

    def test_save_jsonl(self, temp_dir):
        """Test saving data to JSONL file."""
        from src.data.data_loader import save_jsonl

        data = [{"key": "value1"}, {"key": "value2"}]
        filepath = temp_dir / "output.jsonl"

        save_jsonl(data, filepath)

        assert filepath.exists()
        with open(filepath, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_save_creates_parent_dirs(self, temp_dir):
        """Test that save_jsonl creates parent directories."""
        from src.data.data_loader import save_jsonl

        filepath = temp_dir / "nested" / "dir" / "output.jsonl"
        data = [{"test": "data"}]

        save_jsonl(data, filepath)
        assert filepath.exists()

    def test_save_unicode(self, temp_dir):
        """Test saving Unicode content (Korean text)."""
        from src.data.data_loader import save_jsonl, load_jsonl

        data = [{"korean": "안녕하세요"}, {"korean": "감사합니다"}]
        filepath = temp_dir / "unicode.jsonl"

        save_jsonl(data, filepath)
        loaded = load_jsonl(filepath)

        assert loaded[0]["korean"] == "안녕하세요"
        assert loaded[1]["korean"] == "감사합니다"


class TestStreamJsonl:
    """Tests for stream_jsonl function."""

    def test_stream_jsonl(self, sample_jsonl_file):
        """Test streaming JSONL file."""
        from src.data.data_loader import stream_jsonl

        items = list(stream_jsonl(sample_jsonl_file))
        assert len(items) == 2
        assert "messages" in items[0]

    def test_stream_large_file(self, temp_dir):
        """Test streaming a larger file."""
        from src.data.data_loader import stream_jsonl

        filepath = temp_dir / "large.jsonl"
        with open(filepath, "w") as f:
            for i in range(100):
                f.write(json.dumps({"id": i}) + "\n")

        count = 0
        for _ in stream_jsonl(filepath):
            count += 1
        assert count == 100


class TestLoadJsonlAsDataset:
    """Tests for load_jsonl_as_dataset function."""

    def test_returns_dataset(self, sample_jsonl_file):
        """Test that function returns HuggingFace Dataset."""
        from src.data.data_loader import load_jsonl_as_dataset

        dataset = load_jsonl_as_dataset(sample_jsonl_file)
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 2


class TestLoadTrainValTestData:
    """Tests for load_train_val_test_data function."""

    def test_load_with_separate_files(self, sample_train_val_files):
        """Test loading separate train and val files."""
        from src.data.data_loader import load_train_val_test_data

        train_path, val_path = sample_train_val_files
        datasets = load_train_val_test_data(train_path, val_path=val_path)

        assert "train" in datasets
        assert "validation" in datasets
        assert len(datasets["train"]) == 2
        assert len(datasets["validation"]) == 1

    def test_load_with_auto_split(self, sample_jsonl_file):
        """Test auto-splitting train data for validation."""
        from src.data.data_loader import load_train_val_test_data

        datasets = load_train_val_test_data(sample_jsonl_file, val_split=0.5)

        assert "train" in datasets
        assert "validation" in datasets


class TestGetFormattingFunc:
    """Tests for get_formatting_func function."""

    def test_chatml_format(self, sample_messages):
        """Test ChatML formatting."""
        from src.data.data_loader import get_formatting_func

        formatter = get_formatting_func("chatml")
        result = formatter(sample_messages[0])

        assert "text" in result
        assert "<|im_start|>system" in result["text"]
        assert "<|im_end|>" in result["text"]

    def test_instruction_format(self, sample_messages):
        """Test instruction formatting."""
        from src.data.data_loader import get_formatting_func

        formatter = get_formatting_func("instruction")
        result = formatter(sample_messages[0])

        assert "text" in result
        assert "### System:" in result["text"]
        assert "### User:" in result["text"]
        assert "### Assistant:" in result["text"]

    def test_llama_format(self, sample_messages):
        """Test Llama formatting."""
        from src.data.data_loader import get_formatting_func

        formatter = get_formatting_func("llama")
        result = formatter(sample_messages[0])

        assert "text" in result
        assert "<<SYS>>" in result["text"]
        assert "[INST]" in result["text"]

    def test_alpaca_format(self, sample_messages):
        """Test Alpaca formatting."""
        from src.data.data_loader import get_formatting_func

        formatter = get_formatting_func("alpaca")
        result = formatter(sample_messages[0])

        assert "text" in result
        assert "### Instruction:" in result["text"]
        assert "### Response:" in result["text"]

    def test_unknown_format_fallback(self):
        """Test fallback to ChatML for unknown format."""
        from src.data.data_loader import get_formatting_func

        formatter = get_formatting_func("unknown_format")
        assert formatter is not None


class TestFormatPreferencePair:
    """Tests for format_preference_pair function."""

    def test_basic_format(self):
        """Test basic preference pair formatting."""
        from src.data.data_loader import format_preference_pair

        result = format_preference_pair(
            prompt="What is Python?",
            chosen="Python is a programming language.",
            rejected="Python is a snake.",
        )

        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert "<|im_start|>user" in result["prompt"]

    def test_with_system_prompt(self):
        """Test formatting with system prompt."""
        from src.data.data_loader import format_preference_pair

        result = format_preference_pair(
            prompt="What is Python?",
            chosen="Python is a programming language.",
            rejected="Python is a snake.",
            system_prompt="You are a coding assistant.",
        )

        assert "<|im_start|>system" in result["prompt"]


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_valid_dataset(self, sample_jsonl_file):
        """Test validation of valid dataset."""
        from src.data.data_loader import load_jsonl_as_dataset, validate_dataset

        dataset = load_jsonl_as_dataset(sample_jsonl_file)
        is_valid = validate_dataset(dataset, ["messages"])

        assert is_valid is True

    def test_missing_field(self, sample_jsonl_file):
        """Test validation with missing field."""
        from src.data.data_loader import load_jsonl_as_dataset, validate_dataset

        dataset = load_jsonl_as_dataset(sample_jsonl_file)
        is_valid = validate_dataset(dataset, ["nonexistent_field"])

        assert is_valid is False

    def test_empty_dataset(self):
        """Test validation of empty dataset."""
        from src.data.data_loader import validate_dataset

        empty_dataset = Dataset.from_list([])
        is_valid = validate_dataset(empty_dataset, ["any_field"])

        assert is_valid is False


class TestValidateMessagesFormat:
    """Tests for validate_messages_format function."""

    def test_valid_messages(self, sample_messages):
        """Test validation of valid message format."""
        from src.data.data_loader import validate_messages_format

        is_valid = validate_messages_format(sample_messages[0])
        assert is_valid is True

    def test_missing_messages_key(self):
        """Test validation with missing messages key."""
        from src.data.data_loader import validate_messages_format

        is_valid = validate_messages_format({"other": "data"})
        assert is_valid is False

    def test_empty_messages(self):
        """Test validation with empty messages list."""
        from src.data.data_loader import validate_messages_format

        is_valid = validate_messages_format({"messages": []})
        assert is_valid is False

    def test_invalid_role(self):
        """Test validation with invalid role."""
        from src.data.data_loader import validate_messages_format

        sample = {
            "messages": [{"role": "invalid_role", "content": "test"}]
        }
        is_valid = validate_messages_format(sample)
        assert is_valid is False


class TestDeduplicateDataset:
    """Tests for deduplicate_dataset function."""

    def test_removes_duplicates(self, temp_dir):
        """Test that duplicates are removed."""
        from src.data.data_loader import load_jsonl_as_dataset, deduplicate_dataset

        filepath = temp_dir / "dupes.jsonl"
        with open(filepath, "w") as f:
            f.write('{"text": "duplicate"}\n')
            f.write('{"text": "duplicate"}\n')
            f.write('{"text": "unique"}\n')

        dataset = load_jsonl_as_dataset(filepath)
        deduped = deduplicate_dataset(dataset, key_field="text")

        assert len(deduped) == 2
