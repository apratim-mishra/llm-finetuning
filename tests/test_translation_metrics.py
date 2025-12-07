"""
Tests for Translation Metrics
Tests src/evaluation/translation_metrics.py functions.
"""

import pytest


class TestCleanTranslation:
    """Tests for clean_translation function."""

    def test_remove_chatml_tokens(self):
        """Test removal of ChatML tokens."""
        from src.evaluation.translation_metrics import clean_translation

        text = "<|im_start|>assistant\nHello world<|im_end|>"
        cleaned = clean_translation(text)
        assert "<|im_start|>" not in cleaned
        assert "<|im_end|>" not in cleaned

    def test_remove_llama_tokens(self):
        """Test removal of Llama tokens."""
        from src.evaluation.translation_metrics import clean_translation

        text = "[INST] prompt [/INST] Hello world"
        cleaned = clean_translation(text)
        assert "[INST]" not in cleaned
        assert "[/INST]" not in cleaned

    def test_remove_artifacts(self):
        """Test removal of common artifacts."""
        from src.evaluation.translation_metrics import clean_translation

        text = "### Assistant: Hello world"
        cleaned = clean_translation(text)
        assert "### Assistant:" not in cleaned

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        from src.evaluation.translation_metrics import clean_translation

        text = "Hello    world\n\n\ntest"
        cleaned = clean_translation(text)
        assert "  " not in cleaned

    def test_empty_input(self):
        """Test empty input handling."""
        from src.evaluation.translation_metrics import clean_translation

        assert clean_translation("") == ""


class TestExtractTranslationFromResponse:
    """Tests for extract_translation_from_response function."""

    def test_extract_clean_response(self):
        """Test extraction from clean response."""
        from src.evaluation.translation_metrics import extract_translation_from_response

        response = "Hello, how are you?"
        result = extract_translation_from_response(response)
        assert result == "Hello, how are you?"

    def test_remove_prefix(self):
        """Test removal of common prefixes."""
        from src.evaluation.translation_metrics import extract_translation_from_response

        response = "Translation: Hello world"
        result = extract_translation_from_response(response)
        assert result == "Hello world"

    def test_remove_source_echo(self):
        """Test removal of echoed source text."""
        from src.evaluation.translation_metrics import extract_translation_from_response

        response = "안녕하세요 Hello"
        result = extract_translation_from_response(response, source_text="안녕하세요")
        assert "안녕하세요" not in result


class TestComputeLengthRatio:
    """Tests for compute_length_ratio function."""

    def test_equal_length(self):
        """Test ratio when lengths are equal."""
        from src.evaluation.translation_metrics import compute_length_ratio

        predictions = ["Hello world"]
        references = ["Hello world"]

        mean, std = compute_length_ratio(predictions, references)
        assert abs(mean - 1.0) < 0.01

    def test_different_lengths(self):
        """Test ratio with different lengths."""
        from src.evaluation.translation_metrics import compute_length_ratio

        predictions = ["Hi"]
        references = ["Hello world"]

        mean, std = compute_length_ratio(predictions, references)
        assert mean < 1.0

    def test_multiple_samples(self):
        """Test with multiple samples."""
        from src.evaluation.translation_metrics import compute_length_ratio

        predictions = ["Hello", "Hi there"]
        references = ["Hello", "Hi there"]

        mean, std = compute_length_ratio(predictions, references)
        assert mean > 0


class TestTranslationMetricsDataclass:
    """Tests for TranslationMetrics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from src.evaluation.translation_metrics import TranslationMetrics

        metrics = TranslationMetrics(bleu=25.5, chrf=50.0, num_samples=100)
        result = metrics.to_dict()

        assert "bleu" in result
        assert result["bleu"] == 25.5
        assert "num_samples" in result

    def test_excludes_none(self):
        """Test that None values are excluded."""
        from src.evaluation.translation_metrics import TranslationMetrics

        metrics = TranslationMetrics(bleu=25.5, comet=None, num_samples=100)
        result = metrics.to_dict()

        assert "comet" not in result


class TestComputeBleu:
    """Tests for compute_bleu function."""

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_perfect_match(self):
        """Test BLEU with perfect match."""
        from src.evaluation.translation_metrics import compute_bleu

        predictions = ["Hello world"]
        references = [["Hello world"]]

        result = compute_bleu(predictions, references)
        assert "bleu" in result
        assert result["bleu"] == 100.0

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_partial_match(self):
        """Test BLEU with partial match."""
        from src.evaluation.translation_metrics import compute_bleu

        predictions = ["Hello there"]
        references = [["Hello world"]]

        result = compute_bleu(predictions, references)
        assert "bleu" in result
        assert result["bleu"] < 100.0

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_returns_brevity_penalty(self):
        """Test that brevity penalty is returned."""
        from src.evaluation.translation_metrics import compute_bleu

        predictions = ["Hello"]
        references = [["Hello world foo bar"]]

        result = compute_bleu(predictions, references)
        assert "bleu_bp" in result


class TestComputeSentenceBleu:
    """Tests for compute_sentence_bleu function."""

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_sentence_bleu(self):
        """Test sentence-level BLEU."""
        from src.evaluation.translation_metrics import compute_sentence_bleu

        prediction = "Hello world"
        references = ["Hello world"]

        score = compute_sentence_bleu(prediction, references)
        assert score == 100.0


class TestComputeChrf:
    """Tests for compute_chrf function."""

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_chrf_computation(self):
        """Test chrF computation."""
        from src.evaluation.translation_metrics import compute_chrf

        predictions = ["Hello world"]
        references = [["Hello world"]]

        result = compute_chrf(predictions, references)
        assert "chrf" in result
        assert result["chrf"] > 0


class TestComputeTer:
    """Tests for compute_ter function."""

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_ter_computation(self):
        """Test TER computation."""
        from src.evaluation.translation_metrics import compute_ter

        predictions = ["Hello world"]
        references = [["Hello world"]]

        result = compute_ter(predictions, references)
        assert "ter" in result
        # Perfect match should have TER of 0
        assert result["ter"] == 0.0


class TestEvaluateTranslation:
    """Tests for evaluate_translation function."""

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_default_metrics(self):
        """Test evaluation with default metrics."""
        from src.evaluation.translation_metrics import evaluate_translation

        predictions = ["Hello world", "Goodbye"]
        references = ["Hello world", "Farewell"]

        result = evaluate_translation(predictions, references)

        assert "num_samples" in result
        assert "bleu" in result
        assert "chrf" in result

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_custom_metrics(self):
        """Test evaluation with custom metric list."""
        from src.evaluation.translation_metrics import evaluate_translation

        predictions = ["Hello"]
        references = ["Hello"]

        result = evaluate_translation(
            predictions, references,
            metrics=["bleu"]
        )

        assert "bleu" in result
        assert "chrf" not in result

    @pytest.mark.skipif(
        not pytest.importorskip("sacrebleu", reason="sacrebleu not installed"),
        reason="sacrebleu required"
    )
    def test_with_cleaning(self):
        """Test evaluation with output cleaning."""
        from src.evaluation.translation_metrics import evaluate_translation

        predictions = ["<|im_start|>assistant\nHello<|im_end|>"]
        references = ["Hello"]

        result = evaluate_translation(
            predictions, references,
            clean_outputs=True
        )

        assert result["bleu"] == 100.0
