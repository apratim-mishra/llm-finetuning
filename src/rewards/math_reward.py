"""
Math Reward Functions for GRPO Training
Provides reward signals for math reasoning tasks.
Supports GSM8K, MATH, and custom problem formats.
"""

import logging
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class RewardResult:
    """Container for reward computation results."""
    reward: float
    predicted_answer: str
    ground_truth: str
    is_correct: bool
    reasoning_score: float = 0.0
    format_score: float = 0.0


def extract_answer(text: str) -> str:
    """
    Extract final numerical answer from solution text.
    Handles various answer formats from different datasets.

    Args:
        text: Model-generated solution text

    Returns:
        Extracted answer string (empty if not found)
    """
    if not text:
        return ""

    # Priority-ordered patterns for answer extraction
    patterns = [
        # Explicit answer markers
        r"(?:The\s+)?[Aa]nswer\s*(?:is)?:?\s*\$?\s*([+-]?[\d,]+\.?\d*)",
        r"####\s*\$?\s*([+-]?[\d,]+\.?\d*)",  # GSM8K format
        r"\\boxed\{([^}]+)\}",  # LaTeX boxed format (MATH dataset)
        r"\*\*([+-]?[\d,]+\.?\d*)\*\*\s*$",  # Bold answer at end
        # Mathematical conclusions
        r"(?:equals?|=|is)\s*\$?\s*([+-]?[\d,]+\.?\d*)\s*(?:dollars?|%)?\s*$",
        r"(?:result|total|sum|difference|product|quotient)\s+(?:is|=)\s*\$?\s*([+-]?[\d,]+\.?\d*)",
        # Fractions
        r"(?:answer|result)\s*(?:is)?:?\s*([+-]?\d+/\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            answer = match.group(1)
            # Clean the extracted answer
            answer = answer.replace(",", "").replace("$", "").replace(" ", "").strip()
            answer = answer.rstrip(".,;:")
            return answer

    # Fallback: find the last number in the text
    numbers = re.findall(r"[+-]?\d+\.?\d*", text)
    if numbers:
        # Prefer numbers that look like final answers (not intermediate calculations)
        # Skip very small numbers that might be step numbers
        for num in reversed(numbers):
            try:
                val = float(num)
                if abs(val) >= 0.01 or val == 0:  # Accept 0 as valid answer
                    return num.rstrip(".,;:")
            except ValueError:
                continue

    return ""


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from LaTeX \\boxed{} format (common in MATH dataset).

    Args:
        text: LaTeX text with boxed answer

    Returns:
        Content inside \\boxed{}
    """
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return ""


def normalize_answer(answer: str) -> Optional[float]:
    """
    Normalize answer string to float for comparison.
    Handles fractions, percentages, and currency.

    Args:
        answer: Answer string to normalize

    Returns:
        Float value or None if parsing fails
    """
    if not answer:
        return None

    try:
        # Clean the string
        clean = answer.replace(",", "").replace("$", "").replace(" ", "").strip()

        # Handle percentage
        if clean.endswith("%"):
            clean = clean[:-1]
            return float(clean) / 100

        # Handle fractions
        if "/" in clean:
            parts = clean.split("/")
            if len(parts) == 2:
                return float(Fraction(int(parts[0]), int(parts[1])))

        # Handle negative numbers with parentheses
        if clean.startswith("(") and clean.endswith(")"):
            clean = "-" + clean[1:-1]

        return float(clean)

    except (ValueError, AttributeError, ZeroDivisionError):
        return None


def answers_match(
    predicted: str,
    ground_truth: str,
    tolerance: float = 0.01,
    relative_tolerance: float = 0.001,
) -> bool:
    """
    Check if predicted answer matches ground truth.
    Uses both absolute and relative tolerance.

    Args:
        predicted: Predicted answer string
        ground_truth: Ground truth answer string
        tolerance: Absolute tolerance for comparison
        relative_tolerance: Relative tolerance for large numbers

    Returns:
        True if answers match within tolerance
    """
    pred_norm = normalize_answer(predicted)
    true_norm = normalize_answer(ground_truth)

    if pred_norm is None or true_norm is None:
        # String comparison as fallback
        return predicted.strip().lower() == ground_truth.strip().lower()

    # Absolute tolerance check
    if abs(pred_norm - true_norm) < tolerance:
        return True

    # Relative tolerance check for larger numbers
    if true_norm != 0:
        rel_diff = abs(pred_norm - true_norm) / abs(true_norm)
        if rel_diff < relative_tolerance:
            return True

    return False


def check_reasoning_quality(text: str) -> float:
    """
    Score reasoning quality from 0 to 1.
    Evaluates structure, mathematical content, and coherence.

    Args:
        text: Model-generated solution

    Returns:
        Quality score between 0 and 1
    """
    if not text:
        return 0.0

    score = 0.0
    text_lower = text.lower()
    words = text.split()
    word_count = len(words)

    # Step-by-step reasoning indicators (max 0.25)
    step_patterns = [
        r"step\s*\d",
        r"first(?:ly)?[,:\s]",
        r"second(?:ly)?[,:\s]",
        r"third(?:ly)?[,:\s]",
        r"then[,:\s]",
        r"next[,:\s]",
        r"finally[,:\s]",
        r"therefore[,:\s]",
        r"thus[,:\s]",
        r"so[,:\s]",
        r"hence[,:\s]",
        r"let(?:'s|\s+us)",
        r"we (?:have|get|can|need|know|find)",
    ]
    step_matches = sum(1 for p in step_patterns if re.search(p, text_lower))
    score += min(step_matches * 0.05, 0.25)

    # Mathematical expressions (max 0.35)
    math_patterns = [
        r"\d+\s*[+\-*/]\s*\d+",  # Basic operations
        r"\d+\s*[×÷]\s*\d+",  # Unicode operators
        r"\d+\s*=\s*\d+",  # Equations
        r"\(\d+",  # Parenthetical expressions
        r"\d+\s*x\s*\d+",  # Multiplication with x
        r"\d+%",  # Percentages
        r"\$\d+",  # Currency
    ]
    math_matches = sum(len(re.findall(p, text)) for p in math_patterns)
    score += min(math_matches * 0.05, 0.35)

    # Proper conclusion (max 0.2)
    if re.search(r"(?:answer|result|total)\s*(?:is|=|:)", text_lower):
        score += 0.15
    if re.search(r"(?:####|\\boxed|\*\*\d+\*\*)", text):
        score += 0.05

    # Length scoring (max 0.2)
    if word_count < 15:
        score -= 0.1  # Too short
    elif 30 <= word_count <= 200:
        score += 0.15  # Good length
    elif word_count > 500:
        score -= 0.05  # Possibly too verbose

    return max(0.0, min(score, 1.0))


def math_accuracy_reward(
    completions: List[str],
    ground_truths: List[str],
    prompts: Optional[List[str]] = None,
    partial_credit: bool = True,
) -> List[float]:
    """
    Compute rewards for math completions.

    Reward structure:
    - 1.0: Correct answer
    - 0.1-0.5: Wrong answer but good reasoning (if partial_credit=True)
    - 0.0: Wrong answer with poor/no reasoning

    Args:
        completions: Model-generated solutions
        ground_truths: Correct answers
        prompts: Original prompts (optional, for context)
        partial_credit: Give partial credit for good reasoning

    Returns:
        List of reward values
    """
    rewards = []

    for i, (completion, ground_truth) in enumerate(zip(completions, ground_truths)):
        predicted = extract_answer(completion)

        if answers_match(predicted, ground_truth):
            reward = 1.0
        elif partial_credit:
            # Wrong answer - give partial credit for reasoning
            reasoning_score = check_reasoning_quality(completion)
            reward = reasoning_score * 0.5  # Scale to max 0.5 for wrong answer
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards


def math_binary_reward(
    completions: List[str],
    ground_truths: List[str],
    prompts: Optional[List[str]] = None,
) -> List[float]:
    """
    Simple binary reward: 1.0 if correct, 0.0 otherwise.
    Use for strict accuracy optimization.

    Args:
        completions: Model-generated solutions
        ground_truths: Correct answers
        prompts: Original prompts (optional)

    Returns:
        List of binary reward values
    """
    rewards = []

    for completion, ground_truth in zip(completions, ground_truths):
        predicted = extract_answer(completion)
        reward = 1.0 if answers_match(predicted, ground_truth) else 0.0
        rewards.append(reward)

    return rewards


def math_format_reward(
    completions: List[str],
    ground_truths: Optional[List[str]] = None,
    prompts: Optional[List[str]] = None,
    target_format: str = "answer",
) -> List[float]:
    """
    Reward for following correct answer format.

    Args:
        completions: Model-generated solutions
        ground_truths: Not used, for API compatibility
        prompts: Not used, for API compatibility
        target_format: Expected format ("answer", "boxed", "gsm8k")

    Returns:
        List of format rewards (0-0.3)
    """
    rewards = []

    format_patterns = {
        "answer": r"[Aa]nswer:?\s*[+-]?[\d,]+\.?\d*",
        "boxed": r"\\boxed\{[^}]+\}",
        "gsm8k": r"####\s*[+-]?[\d,]+\.?\d*",
    }

    pattern = format_patterns.get(target_format, format_patterns["answer"])

    for completion in completions:
        if re.search(pattern, completion):
            reward = 0.3
        elif re.search(r"(?:answer|result|total)\s*(?:is|=|:)", completion, re.I):
            reward = 0.15  # Partial credit for semi-proper format
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards


def combined_math_reward(
    completions: List[str],
    ground_truths: List[str],
    prompts: Optional[List[str]] = None,
    accuracy_weight: float = 0.7,
    format_weight: float = 0.15,
    reasoning_weight: float = 0.15,
) -> List[float]:
    """
    Combined reward: accuracy + format + reasoning quality.

    Args:
        completions: Model-generated solutions
        ground_truths: Correct answers
        prompts: Original prompts (optional)
        accuracy_weight: Weight for correctness (default 0.7)
        format_weight: Weight for format compliance (default 0.15)
        reasoning_weight: Weight for reasoning quality (default 0.15)

    Returns:
        List of combined reward values
    """
    accuracy_rewards = math_accuracy_reward(
        completions, ground_truths, prompts, partial_credit=False
    )
    format_rewards = math_format_reward(completions)
    reasoning_rewards = [check_reasoning_quality(c) for c in completions]

    combined = []
    for acc, fmt, reason in zip(accuracy_rewards, format_rewards, reasoning_rewards):
        # If correct, still add format and reasoning bonuses
        reward = (
            accuracy_weight * acc
            + format_weight * (fmt / 0.3)  # Normalize format to 0-1
            + reasoning_weight * reason
        )
        combined.append(min(reward, 1.0))  # Cap at 1.0

    return combined


def get_reward_function(name: str) -> Callable:
    """
    Get a reward function by name.

    Args:
        name: Reward function name

    Returns:
        Reward function callable

    Available functions:
        - "accuracy": math_accuracy_reward (with partial credit)
        - "binary": math_binary_reward (strict 0/1)
        - "format": math_format_reward
        - "combined": combined_math_reward
    """
    functions = {
        "accuracy": math_accuracy_reward,
        "math_accuracy_reward": math_accuracy_reward,
        "binary": math_binary_reward,
        "math_binary_reward": math_binary_reward,
        "format": math_format_reward,
        "math_format_reward": math_format_reward,
        "combined": combined_math_reward,
        "combined_math_reward": combined_math_reward,
    }

    if name not in functions:
        logger.warning(f"Unknown reward function '{name}', using 'accuracy'")
        return math_accuracy_reward

    return functions[name]


def compute_accuracy(
    completions: List[str],
    ground_truths: List[str],
) -> Tuple[float, int, int]:
    """
    Compute accuracy statistics.

    Args:
        completions: Model predictions
        ground_truths: Correct answers

    Returns:
        Tuple of (accuracy, num_correct, total)
    """
    correct = 0
    total = len(completions)

    for completion, gt in zip(completions, ground_truths):
        predicted = extract_answer(completion)
        if answers_match(predicted, gt):
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def evaluate_math_batch(
    completions: List[str],
    ground_truths: List[str],
    prompts: Optional[List[str]] = None,
) -> Dict:
    """
    Comprehensive evaluation of a batch of math solutions.

    Args:
        completions: Model-generated solutions
        ground_truths: Correct answers
        prompts: Original problems (optional)

    Returns:
        Dict with accuracy, reward statistics, and per-sample results
    """
    accuracy, correct, total = compute_accuracy(completions, ground_truths)
    rewards = math_accuracy_reward(completions, ground_truths, prompts)
    reasoning_scores = [check_reasoning_quality(c) for c in completions]

    results = []
    for i, (comp, gt) in enumerate(zip(completions, ground_truths)):
        pred = extract_answer(comp)
        results.append(
            RewardResult(
                reward=rewards[i],
                predicted_answer=pred,
                ground_truth=gt,
                is_correct=answers_match(pred, gt),
                reasoning_score=reasoning_scores[i],
            )
        )

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0,
        "mean_reasoning_score": round(sum(reasoning_scores) / len(reasoning_scores), 4)
        if reasoning_scores
        else 0,
        "results": results,
    }
