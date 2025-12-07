"""
Tests for Math Reward Functions
Tests src/rewards/math_reward.py functions.
"""

import pytest


class TestExtractAnswer:
    """Tests for extract_answer function."""

    def test_extract_gsm8k_format(self):
        """Test extraction from GSM8K format (####)."""
        from src.rewards.math_reward import extract_answer

        text = "Let's solve this step by step. 5 + 3 = 8. #### 8"
        answer = extract_answer(text)
        assert answer == "8"

    def test_extract_answer_is_format(self):
        """Test extraction from 'Answer is X' format."""
        from src.rewards.math_reward import extract_answer

        text = "Therefore, the answer is 42."
        answer = extract_answer(text)
        assert answer == "42"

    def test_extract_boxed_format(self):
        """Test extraction from LaTeX boxed format."""
        from src.rewards.math_reward import extract_answer

        text = "The solution is \\boxed{123}."
        answer = extract_answer(text)
        assert answer == "123"

    def test_extract_with_dollar_sign(self):
        """Test extraction with currency."""
        from src.rewards.math_reward import extract_answer

        text = "The total cost is $150."
        answer = extract_answer(text)
        assert answer == "150"

    def test_extract_with_commas(self):
        """Test extraction with number containing commas."""
        from src.rewards.math_reward import extract_answer

        text = "The answer is 1,234,567."
        answer = extract_answer(text)
        assert answer == "1234567"

    def test_extract_decimal(self):
        """Test extraction of decimal number."""
        from src.rewards.math_reward import extract_answer

        text = "The result is 3.14159."
        answer = extract_answer(text)
        assert answer == "3.14159"

    def test_extract_negative(self):
        """Test extraction of negative number."""
        from src.rewards.math_reward import extract_answer

        text = "The answer is -42."
        answer = extract_answer(text)
        assert answer == "-42"

    def test_extract_fallback_last_number(self):
        """Test fallback to last number in text."""
        from src.rewards.math_reward import extract_answer

        text = "First we calculate 5, then 10, finally 15."
        answer = extract_answer(text)
        assert answer == "15"

    def test_extract_empty_text(self):
        """Test extraction from empty text."""
        from src.rewards.math_reward import extract_answer

        assert extract_answer("") == ""
        assert extract_answer(None) == ""


class TestExtractBoxedAnswer:
    """Tests for extract_boxed_answer function."""

    def test_simple_boxed(self):
        """Test simple boxed extraction."""
        from src.rewards.math_reward import extract_boxed_answer

        text = "Therefore \\boxed{42}"
        answer = extract_boxed_answer(text)
        assert answer == "42"

    def test_boxed_with_expression(self):
        """Test boxed with mathematical expression."""
        from src.rewards.math_reward import extract_boxed_answer

        text = "The answer is \\boxed{2x + 3}"
        answer = extract_boxed_answer(text)
        assert answer == "2x + 3"

    def test_no_boxed(self):
        """Test when no boxed content exists."""
        from src.rewards.math_reward import extract_boxed_answer

        text = "No boxed answer here"
        answer = extract_boxed_answer(text)
        assert answer == ""


class TestNormalizeAnswer:
    """Tests for normalize_answer function."""

    def test_normalize_integer(self):
        """Test normalizing integer."""
        from src.rewards.math_reward import normalize_answer

        assert normalize_answer("42") == 42.0

    def test_normalize_decimal(self):
        """Test normalizing decimal."""
        from src.rewards.math_reward import normalize_answer

        assert abs(normalize_answer("3.14") - 3.14) < 0.001

    def test_normalize_with_commas(self):
        """Test normalizing number with commas."""
        from src.rewards.math_reward import normalize_answer

        assert normalize_answer("1,234,567") == 1234567.0

    def test_normalize_percentage(self):
        """Test normalizing percentage."""
        from src.rewards.math_reward import normalize_answer

        result = normalize_answer("50%")
        assert abs(result - 0.5) < 0.001

    def test_normalize_fraction(self):
        """Test normalizing fraction."""
        from src.rewards.math_reward import normalize_answer

        result = normalize_answer("1/2")
        assert abs(result - 0.5) < 0.001

    def test_normalize_currency(self):
        """Test normalizing currency."""
        from src.rewards.math_reward import normalize_answer

        assert normalize_answer("$100") == 100.0

    def test_normalize_invalid(self):
        """Test normalizing invalid input."""
        from src.rewards.math_reward import normalize_answer

        assert normalize_answer("") is None
        assert normalize_answer("not a number") is None


class TestAnswersMatch:
    """Tests for answers_match function."""

    def test_exact_match(self):
        """Test exact numerical match."""
        from src.rewards.math_reward import answers_match

        assert answers_match("42", "42") is True

    def test_within_tolerance(self):
        """Test match within absolute tolerance."""
        from src.rewards.math_reward import answers_match

        assert answers_match("3.141", "3.14159") is True

    def test_relative_tolerance(self):
        """Test match within relative tolerance for large numbers."""
        from src.rewards.math_reward import answers_match

        assert answers_match("1000000", "1000001") is True

    def test_mismatch(self):
        """Test numerical mismatch."""
        from src.rewards.math_reward import answers_match

        assert answers_match("10", "20") is False

    def test_string_fallback(self):
        """Test string comparison fallback."""
        from src.rewards.math_reward import answers_match

        assert answers_match("abc", "abc") is True
        assert answers_match("abc", "def") is False


class TestCheckReasoningQuality:
    """Tests for check_reasoning_quality function."""

    def test_high_quality_reasoning(self):
        """Test scoring of high-quality reasoning."""
        from src.rewards.math_reward import check_reasoning_quality

        text = """
        Let's solve this step by step.
        First, we need to find the total: 5 + 3 = 8
        Then, we multiply by 2: 8 * 2 = 16
        Therefore, the answer is 16.
        """
        score = check_reasoning_quality(text)
        assert score > 0.5

    def test_low_quality_reasoning(self):
        """Test scoring of low-quality reasoning."""
        from src.rewards.math_reward import check_reasoning_quality

        text = "42"
        score = check_reasoning_quality(text)
        assert score < 0.3

    def test_empty_text(self):
        """Test scoring of empty text."""
        from src.rewards.math_reward import check_reasoning_quality

        assert check_reasoning_quality("") == 0.0

    def test_medium_quality_reasoning(self):
        """Test scoring of medium-quality reasoning."""
        from src.rewards.math_reward import check_reasoning_quality

        text = "We have 5 + 3 = 8. The answer is 8."
        score = check_reasoning_quality(text)
        assert 0.1 <= score <= 0.8


class TestMathAccuracyReward:
    """Tests for math_accuracy_reward function."""

    def test_correct_answer(self):
        """Test reward for correct answer."""
        from src.rewards.math_reward import math_accuracy_reward

        completions = ["After calculation, the answer is 42"]
        ground_truths = ["42"]

        rewards = math_accuracy_reward(completions, ground_truths)
        assert rewards[0] == 1.0

    def test_wrong_answer_with_partial_credit(self):
        """Test partial credit for wrong answer with good reasoning."""
        from src.rewards.math_reward import math_accuracy_reward

        completions = [
            "Let's solve step by step. First, 5 + 3 = 8. Then 8 * 2 = 16. The answer is 16."
        ]
        ground_truths = ["20"]

        rewards = math_accuracy_reward(completions, ground_truths, partial_credit=True)
        assert 0.0 < rewards[0] < 1.0

    def test_wrong_answer_no_partial_credit(self):
        """Test no partial credit when disabled."""
        from src.rewards.math_reward import math_accuracy_reward

        completions = ["The answer is 100"]
        ground_truths = ["42"]

        rewards = math_accuracy_reward(completions, ground_truths, partial_credit=False)
        assert rewards[0] == 0.0

    def test_multiple_completions(self):
        """Test multiple completions."""
        from src.rewards.math_reward import math_accuracy_reward

        completions = ["The answer is 42", "The answer is 100"]
        ground_truths = ["42", "42"]

        rewards = math_accuracy_reward(completions, ground_truths)
        assert rewards[0] == 1.0
        assert rewards[1] < 1.0


class TestMathBinaryReward:
    """Tests for math_binary_reward function."""

    def test_correct_gives_1(self):
        """Test correct answer gives 1.0."""
        from src.rewards.math_reward import math_binary_reward

        completions = ["The answer is 42"]
        ground_truths = ["42"]

        rewards = math_binary_reward(completions, ground_truths)
        assert rewards[0] == 1.0

    def test_wrong_gives_0(self):
        """Test wrong answer gives 0.0."""
        from src.rewards.math_reward import math_binary_reward

        completions = ["The answer is 100"]
        ground_truths = ["42"]

        rewards = math_binary_reward(completions, ground_truths)
        assert rewards[0] == 0.0


class TestMathFormatReward:
    """Tests for math_format_reward function."""

    def test_answer_format(self):
        """Test reward for proper answer format."""
        from src.rewards.math_reward import math_format_reward

        completions = ["Answer: 42"]
        rewards = math_format_reward(completions, target_format="answer")
        assert rewards[0] == 0.3

    def test_gsm8k_format(self):
        """Test reward for GSM8K format."""
        from src.rewards.math_reward import math_format_reward

        completions = ["Step by step... #### 42"]
        rewards = math_format_reward(completions, target_format="gsm8k")
        assert rewards[0] == 0.3

    def test_boxed_format(self):
        """Test reward for boxed format."""
        from src.rewards.math_reward import math_format_reward

        completions = ["Therefore \\boxed{42}"]
        rewards = math_format_reward(completions, target_format="boxed")
        assert rewards[0] == 0.3

    def test_partial_format(self):
        """Test partial reward for semi-proper format."""
        from src.rewards.math_reward import math_format_reward

        completions = ["The result is 42"]
        rewards = math_format_reward(completions)
        assert rewards[0] == 0.15

    def test_no_format(self):
        """Test zero reward for no proper format."""
        from src.rewards.math_reward import math_format_reward

        completions = ["42"]
        rewards = math_format_reward(completions)
        assert rewards[0] == 0.0


class TestCombinedMathReward:
    """Tests for combined_math_reward function."""

    def test_correct_answer_high_reward(self):
        """Test correct answer with good format gets high reward."""
        from src.rewards.math_reward import combined_math_reward

        completions = [
            "Let's solve step by step. First 5 + 3 = 8. Answer: 8"
        ]
        ground_truths = ["8"]

        rewards = combined_math_reward(completions, ground_truths)
        assert rewards[0] > 0.8

    def test_custom_weights(self):
        """Test custom weight configuration."""
        from src.rewards.math_reward import combined_math_reward

        completions = ["Answer: 42"]
        ground_truths = ["42"]

        rewards = combined_math_reward(
            completions,
            ground_truths,
            accuracy_weight=0.5,
            format_weight=0.25,
            reasoning_weight=0.25,
        )
        assert len(rewards) == 1


class TestGetRewardFunction:
    """Tests for get_reward_function function."""

    def test_get_accuracy(self):
        """Test getting accuracy reward function."""
        from src.rewards.math_reward import get_reward_function, math_accuracy_reward

        func = get_reward_function("accuracy")
        assert func == math_accuracy_reward

    def test_get_binary(self):
        """Test getting binary reward function."""
        from src.rewards.math_reward import get_reward_function, math_binary_reward

        func = get_reward_function("binary")
        assert func == math_binary_reward

    def test_get_combined(self):
        """Test getting combined reward function."""
        from src.rewards.math_reward import get_reward_function, combined_math_reward

        func = get_reward_function("combined")
        assert func == combined_math_reward

    def test_unknown_falls_back(self):
        """Test unknown name falls back to accuracy."""
        from src.rewards.math_reward import get_reward_function, math_accuracy_reward

        func = get_reward_function("unknown")
        assert func == math_accuracy_reward


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""

    def test_all_correct(self):
        """Test accuracy when all correct."""
        from src.rewards.math_reward import compute_accuracy

        completions = ["Answer: 10", "Answer: 20"]
        ground_truths = ["10", "20"]

        accuracy, correct, total = compute_accuracy(completions, ground_truths)
        assert accuracy == 1.0
        assert correct == 2
        assert total == 2

    def test_none_correct(self):
        """Test accuracy when none correct."""
        from src.rewards.math_reward import compute_accuracy

        completions = ["Answer: 10", "Answer: 20"]
        ground_truths = ["100", "200"]

        accuracy, correct, total = compute_accuracy(completions, ground_truths)
        assert accuracy == 0.0
        assert correct == 0


class TestEvaluateMathBatch:
    """Tests for evaluate_math_batch function."""

    def test_batch_evaluation(self):
        """Test batch evaluation returns proper structure."""
        from src.rewards.math_reward import evaluate_math_batch

        completions = ["Answer: 10", "Answer: 20"]
        ground_truths = ["10", "25"]

        result = evaluate_math_batch(completions, ground_truths)

        assert "accuracy" in result
        assert "correct" in result
        assert "total" in result
        assert "mean_reward" in result
        assert "results" in result
        assert result["correct"] == 1
        assert result["total"] == 2
