"""
Data Loading Utilities
Shared between MLX and GPU training pipelines.
Supports ChatML, instruction, and preference pair formats.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


def load_jsonl(filepath: Union[str, Path]) -> list[dict]:
    """
    Load JSONL file into list of dicts.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of dictionaries from JSONL
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")

    logger.info(f"Loaded {len(data)} samples from {filepath}")
    return data


def save_jsonl(data: list[dict], filepath: Union[str, Path]) -> None:
    """
    Save list of dicts to JSONL file.

    Args:
        data: List of dictionaries to save
        filepath: Output path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(data)} samples to {filepath}")


def stream_jsonl(filepath: Union[str, Path]) -> Iterator[dict]:
    """
    Stream JSONL file line by line for memory efficiency.

    Args:
        filepath: Path to JSONL file

    Yields:
        Dictionaries from JSONL one at a time
    """
    filepath = Path(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_jsonl_as_dataset(filepath: Union[str, Path]) -> Dataset:
    """
    Load JSONL file as HuggingFace Dataset.

    Args:
        filepath: Path to JSONL file

    Returns:
        HuggingFace Dataset
    """
    data = load_jsonl(filepath)
    return Dataset.from_list(data)


def load_train_val_test_data(
    train_path: Union[str, Path],
    val_path: Optional[Union[str, Path]] = None,
    test_path: Optional[Union[str, Path]] = None,
    val_split: float = 0.1,
    test_split: float = 0.0,
    seed: int = 42,
) -> DatasetDict:
    """
    Load training, validation, and test datasets.
    Creates splits from train data if paths not provided.

    Args:
        train_path: Path to training JSONL
        val_path: Optional path to validation JSONL
        test_path: Optional path to test JSONL
        val_split: Fraction for validation if not provided
        test_split: Fraction for test if not provided
        seed: Random seed for splitting

    Returns:
        DatasetDict with train, validation, and optionally test splits
    """
    datasets = {}
    train_data = load_jsonl_as_dataset(train_path)

    # Handle validation split
    if val_path and Path(val_path).exists():
        datasets["train"] = train_data
        datasets["validation"] = load_jsonl_as_dataset(val_path)
    elif val_split > 0:
        split = train_data.train_test_split(test_size=val_split, seed=seed)
        datasets["train"] = split["train"]
        datasets["validation"] = split["test"]
    else:
        datasets["train"] = train_data

    # Handle test split
    if test_path and Path(test_path).exists():
        datasets["test"] = load_jsonl_as_dataset(test_path)
    elif test_split > 0 and "train" in datasets:
        split = datasets["train"].train_test_split(test_size=test_split, seed=seed)
        datasets["train"] = split["train"]
        datasets["test"] = split["test"]

    return DatasetDict(datasets)


def get_formatting_func(format_type: str = "chatml") -> Callable[[dict], dict]:
    """
    Get formatting function for different conversation formats.

    Args:
        format_type: One of "chatml", "instruction", "llama", "alpaca"

    Returns:
        Function that converts sample dict to formatted text dict
    """

    def chatml_format(sample: dict) -> dict:
        """Format messages to ChatML text (Qwen, Mistral style)."""
        messages = sample.get("messages", [])
        text_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return {"text": "\n".join(text_parts)}

    def instruction_format(sample: dict) -> dict:
        """Format as ### instruction/response style."""
        messages = sample.get("messages", [])
        text_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text_parts.append(f"### System:\n{content}")
            elif role == "user":
                text_parts.append(f"### User:\n{content}")
            elif role == "assistant":
                text_parts.append(f"### Assistant:\n{content}")

        return {"text": "\n\n".join(text_parts)}

    def llama_format(sample: dict) -> dict:
        """Format for Llama-style chat template."""
        messages = sample.get("messages", [])
        text_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text_parts.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                text_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                text_parts.append(content)

        return {"text": "\n".join(text_parts)}

    def alpaca_format(sample: dict) -> dict:
        """Format for Alpaca-style instruction tuning."""
        messages = sample.get("messages", [])
        system = ""
        instruction = ""
        response = ""

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            elif msg["role"] == "user":
                instruction = msg["content"]
            elif msg["role"] == "assistant":
                response = msg["content"]

        text = f"### Instruction:\n{instruction}\n\n"
        if system:
            text = f"### System:\n{system}\n\n" + text
        text += f"### Response:\n{response}"

        return {"text": text}

    formatters = {
        "chatml": chatml_format,
        "instruction": instruction_format,
        "llama": llama_format,
        "alpaca": alpaca_format,
    }

    if format_type not in formatters:
        logger.warning(f"Unknown format '{format_type}', using chatml")
        return formatters["chatml"]

    return formatters[format_type]


def format_preference_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    system_prompt: Optional[str] = None,
) -> dict:
    """
    Format a preference pair for DPO training.

    Args:
        prompt: User query/instruction
        chosen: Preferred response
        rejected: Non-preferred response
        system_prompt: Optional system message

    Returns:
        Dict with prompt, chosen, rejected keys
    """
    if system_prompt:
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        full_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    return {
        "prompt": full_prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def validate_dataset(
    dataset: Dataset,
    required_fields: list[str],
    raise_on_error: bool = False,
) -> bool:
    """
    Validate dataset has required fields.

    Args:
        dataset: HuggingFace Dataset to validate
        required_fields: List of required field names
        raise_on_error: Raise exception instead of returning False

    Returns:
        True if valid, False otherwise
    """
    if len(dataset) == 0:
        msg = "Dataset is empty"
        if raise_on_error:
            raise ValueError(msg)
        logger.error(msg)
        return False

    sample = dataset[0]
    missing = [f for f in required_fields if f not in sample]

    if missing:
        msg = f"Missing required fields: {missing}"
        if raise_on_error:
            raise ValueError(msg)
        logger.error(msg)
        return False

    return True


def validate_messages_format(sample: dict) -> bool:
    """
    Validate a sample has proper ChatML message format.

    Args:
        sample: Dict with 'messages' key

    Returns:
        True if valid format
    """
    if "messages" not in sample:
        return False

    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return False

    valid_roles = {"system", "user", "assistant"}
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in valid_roles:
            return False

    return True


def filter_by_length(
    dataset: Dataset,
    max_length: int,
    text_field: str = "text",
    tokenizer=None,
) -> Dataset:
    """
    Filter dataset by sequence length.

    Args:
        dataset: Input dataset
        max_length: Maximum token/character length
        text_field: Field containing text
        tokenizer: Optional tokenizer for token counting

    Returns:
        Filtered dataset
    """
    def length_filter(sample):
        text = sample.get(text_field, "")
        if tokenizer:
            return len(tokenizer.encode(text)) <= max_length
        return len(text) <= max_length * 4  # Rough char estimate

    original_len = len(dataset)
    filtered = dataset.filter(length_filter)
    logger.info(f"Filtered {original_len - len(filtered)} samples exceeding length {max_length}")

    return filtered


def deduplicate_dataset(
    dataset: Dataset,
    key_field: str = "text",
) -> Dataset:
    """
    Remove duplicate entries from dataset.

    Args:
        dataset: Input dataset
        key_field: Field to use for deduplication

    Returns:
        Deduplicated dataset
    """
    seen = set()
    indices_to_keep = []

    for i, sample in enumerate(dataset):
        key = sample.get(key_field, "")
        if key not in seen:
            seen.add(key)
            indices_to_keep.append(i)

    original_len = len(dataset)
    deduped = dataset.select(indices_to_keep)
    logger.info(f"Removed {original_len - len(deduped)} duplicate samples")

    return deduped
