"""
Translation Metrics
BLEU, COMET, chrF, TER evaluation for Korean-English translation.
Supports sentence-level and corpus-level evaluation.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TranslationMetrics:
    """Container for translation evaluation results."""
    bleu: Optional[float] = None
    bleu_bp: Optional[float] = None
    chrf: Optional[float] = None
    ter: Optional[float] = None
    comet: Optional[float] = None
    num_samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


def compute_bleu(
    predictions: List[str],
    references: List[List[str]],
    tokenize: str = "13a",
    lowercase: bool = False,
) -> Dict[str, float]:
    """
    Compute BLEU score using sacrebleu.

    Args:
        predictions: List of predicted translations
        references: List of reference translations (list of lists for multi-ref)
        tokenize: Tokenization scheme (13a, intl, zh, ja-mecab, ko-mecab)
        lowercase: Whether to lowercase before scoring

    Returns:
        Dict with BLEU score and details
    """
    try:
        from sacrebleu.metrics import BLEU

        bleu = BLEU(tokenize=tokenize, lowercase=lowercase, effective_order=True)
        result = bleu.corpus_score(predictions, references)

        return {
            "bleu": round(result.score, 2),
            "bleu_bp": round(result.bp, 4),
            "bleu_precisions": [round(p, 2) for p in result.precisions],
            "bleu_details": str(result),
        }
    except ImportError:
        logger.error("sacrebleu not installed. Run: pip install sacrebleu")
        return {"error": "sacrebleu not installed"}
    except Exception as e:
        logger.error(f"BLEU computation failed: {e}")
        return {"error": str(e)}


def compute_sentence_bleu(
    prediction: str,
    references: List[str],
    tokenize: str = "13a",
) -> float:
    """
    Compute sentence-level BLEU score.

    Args:
        prediction: Single predicted translation
        references: List of reference translations

    Returns:
        Sentence BLEU score
    """
    try:
        from sacrebleu.metrics import BLEU

        bleu = BLEU(tokenize=tokenize, effective_order=True)
        result = bleu.sentence_score(prediction, references)
        return round(result.score, 2)
    except ImportError:
        return 0.0


def compute_chrf(
    predictions: List[str],
    references: List[List[str]],
    word_order: int = 0,
    char_order: int = 6,
    beta: float = 2.0,
) -> Dict[str, float]:
    """
    Compute chrF score (character-level F-score).
    Particularly good for morphologically rich languages like Korean.

    Args:
        predictions: Predicted translations
        references: Reference translations
        word_order: Word n-gram order (0 for character-only)
        char_order: Character n-gram order
        beta: Recall weight (2.0 emphasizes recall)

    Returns:
        Dict with chrF score and details
    """
    try:
        from sacrebleu.metrics import CHRF

        chrf = CHRF(word_order=word_order, char_order=char_order, beta=beta)
        result = chrf.corpus_score(predictions, references)

        return {
            "chrf": round(result.score, 2),
            "chrf_details": str(result),
        }
    except ImportError:
        logger.error("sacrebleu not installed")
        return {"error": "sacrebleu not installed"}
    except Exception as e:
        return {"error": str(e)}


def compute_comet(
    predictions: List[str],
    references: List[str],
    sources: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 0,
) -> Dict[str, float]:
    """
    Compute COMET score (neural semantic similarity).
    Requires source sentences for context.

    Args:
        predictions: Predicted translations
        references: Reference translations (flat list)
        sources: Source sentences
        model_name: COMET model to use
        batch_size: Batch size for inference
        gpus: Number of GPUs (0 for CPU)

    Returns:
        Dict with COMET score and per-sentence scores

    Note:
        COMET model download is ~2GB on first use.
    """
    try:
        from comet import download_model, load_from_checkpoint

        logger.info(f"Loading COMET model: {model_name}")
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)

        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]

        output = model.predict(data, batch_size=batch_size, gpus=gpus)

        return {
            "comet": round(output.system_score, 4),
            "comet_scores": [round(s, 4) for s in output.scores],
        }
    except ImportError:
        logger.warning("unbabel-comet not installed. Run: pip install unbabel-comet")
        return {"error": "unbabel-comet not installed"}
    except Exception as e:
        logger.error(f"COMET computation failed: {e}")
        return {"error": str(e)}


def compute_ter(
    predictions: List[str],
    references: List[List[str]],
    normalized: bool = False,
    asian_support: bool = True,
) -> Dict[str, float]:
    """
    Compute Translation Edit Rate (TER).
    Lower scores are better.

    Args:
        predictions: Predicted translations
        references: Reference translations
        normalized: Apply normalization
        asian_support: Better handling of Asian languages

    Returns:
        Dict with TER score
    """
    try:
        from sacrebleu.metrics import TER

        ter = TER(normalized=normalized, asian_support=asian_support)
        result = ter.corpus_score(predictions, references)

        return {
            "ter": round(result.score, 2),
            "ter_details": str(result),
        }
    except ImportError:
        logger.error("sacrebleu not installed")
        return {"error": "sacrebleu not installed"}
    except Exception as e:
        return {"error": str(e)}


def compute_bleurt(
    predictions: List[str],
    references: List[str],
    checkpoint: str = "BLEURT-20",
) -> Dict[str, float]:
    """
    Compute BLEURT score (learned metric, correlates well with human judgment).

    Args:
        predictions: Predicted translations
        references: Reference translations (flat list)
        checkpoint: BLEURT checkpoint name

    Returns:
        Dict with BLEURT score and per-sentence scores

    Note:
        BLEURT model download is ~2GB on first use.
        Install with: pip install bleurt @ git+https://github.com/google-research/bleurt.git
    """
    try:
        from bleurt import score as bleurt_score

        logger.info(f"Loading BLEURT checkpoint: {checkpoint}")
        scorer = bleurt_score.BleurtScorer(checkpoint)
        scores = scorer.score(references=references, candidates=predictions)

        return {
            "bleurt": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "bleurt_scores": [round(s, 4) for s in scores],
        }
    except ImportError:
        logger.warning(
            "BLEURT not installed. Install with: "
            "pip install bleurt @ git+https://github.com/google-research/bleurt.git"
        )
        return {"error": "bleurt not installed"}
    except Exception as e:
        logger.error(f"BLEURT computation failed: {e}")
        return {"error": str(e)}


def compute_xcomet(
    predictions: List[str],
    references: List[str],
    sources: List[str],
    model_name: str = "Unbabel/XCOMET-XL",
    batch_size: int = 8,
    gpus: int = 0,
) -> Dict[str, float]:
    """
    Compute xCOMET score (state-of-the-art neural MT metric).
    xCOMET is an improved version of COMET with better correlation to human judgment.

    Args:
        predictions: Predicted translations
        references: Reference translations (flat list)
        sources: Source sentences
        model_name: xCOMET model to use (XCOMET-XL or XCOMET-XXL)
        batch_size: Batch size for inference
        gpus: Number of GPUs (0 for CPU)

    Returns:
        Dict with xCOMET score and per-sentence scores

    Note:
        xCOMET model download is ~3GB on first use.
    """
    try:
        from comet import download_model, load_from_checkpoint

        logger.info(f"Loading xCOMET model: {model_name}")
        model_path = download_model(model_name)
        model = load_from_checkpoint(model_path)

        data = [
            {"src": src, "mt": pred, "ref": ref}
            for src, pred, ref in zip(sources, predictions, references)
        ]

        output = model.predict(data, batch_size=batch_size, gpus=gpus)

        return {
            "xcomet": round(output.system_score, 4),
            "xcomet_scores": [round(s, 4) for s in output.scores],
        }
    except ImportError:
        logger.warning("unbabel-comet not installed. Run: pip install unbabel-comet>=2.2")
        return {"error": "unbabel-comet not installed"}
    except Exception as e:
        logger.error(f"xCOMET computation failed: {e}")
        return {"error": str(e)}


def evaluate_translation(
    predictions: List[str],
    references: List[str],
    sources: Optional[List[str]] = None,
    metrics: List[str] = None,
    clean_outputs: bool = True,
) -> Dict[str, float]:
    """
    Compute multiple translation metrics.

    Args:
        predictions: Translated texts
        references: Reference translations
        sources: Source texts (required for COMET, xCOMET)
        metrics: Metrics to compute (default: bleu, chrf)
            Available: bleu, chrf, ter, comet, xcomet, bleurt
        clean_outputs: Clean predictions before scoring

    Returns:
        Dict with all computed metric scores
    """
    if metrics is None:
        metrics = ["bleu", "chrf"]

    # Clean predictions if requested
    if clean_outputs:
        predictions = [clean_translation(p) for p in predictions]
        references = [clean_translation(r) for r in references]

    # Format references for sacrebleu (list of lists for multi-reference)
    refs_formatted = [[ref] for ref in references]

    results = {
        "num_samples": len(predictions),
    }

    if "bleu" in metrics:
        bleu_result = compute_bleu(predictions, refs_formatted)
        results.update(bleu_result)

    if "chrf" in metrics:
        chrf_result = compute_chrf(predictions, refs_formatted)
        results.update(chrf_result)

    if "ter" in metrics:
        ter_result = compute_ter(predictions, refs_formatted)
        results.update(ter_result)

    if "comet" in metrics:
        if sources is not None:
            comet_result = compute_comet(predictions, references, sources)
            results.update(comet_result)
        else:
            logger.warning("COMET requires source sentences, skipping")

    if "xcomet" in metrics:
        if sources is not None:
            xcomet_result = compute_xcomet(predictions, references, sources)
            results.update(xcomet_result)
        else:
            logger.warning("xCOMET requires source sentences, skipping")

    if "bleurt" in metrics:
        bleurt_result = compute_bleurt(predictions, references)
        results.update(bleurt_result)

    return results


def clean_translation(text: str) -> str:
    """
    Clean generated translation text by removing special tokens and artifacts.

    Args:
        text: Raw model output

    Returns:
        Cleaned translation text
    """
    if not text:
        return ""

    # Remove ChatML tokens while keeping the inner content
    text = text.replace("<|im_end|>", "")
    text = re.sub(r"<\|im_start\|>\s*(system|user|assistant)\s*\n", "", text)
    text = text.replace("<|im_start|>", "")
    text = re.sub(r"<\|[^>]*\|>", "", text)

    # Remove Llama tokens
    text = re.sub(r"\[INST\].*?\[/INST\]", "", text, flags=re.DOTALL)
    text = re.sub(r"<<SYS>>.*?<</SYS>>", "", text, flags=re.DOTALL)

    # Remove common artifacts
    artifacts = [
        "### Assistant:",
        "### Response:",
        "assistant",
        "Assistant:",
        "Translation:",
    ]
    for artifact in artifacts:
        text = text.replace(artifact, "")

    # Remove leading/trailing special chars and normalize whitespace
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def extract_translation_from_response(
    response: str,
    source_text: Optional[str] = None,
) -> str:
    """
    Extract just the translation from a model response.

    Args:
        response: Full model response
        source_text: Original source text to avoid including

    Returns:
        Extracted translation
    """
    # First clean the response
    text = clean_translation(response)

    # If source text provided, remove it if echoed
    if source_text:
        text = text.replace(source_text, "").strip()

    # Remove common prefixes
    prefixes = [
        "The translation is:",
        "English translation:",
        "Here is the translation:",
        "Translation:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    return text


def compute_length_ratio(
    predictions: List[str],
    references: List[str],
) -> Tuple[float, float]:
    """
    Compute length ratio statistics.

    Args:
        predictions: Predicted translations
        references: Reference translations

    Returns:
        Tuple of (mean_ratio, std_ratio)
    """
    import statistics

    ratios = []
    for pred, ref in zip(predictions, references):
        if len(ref) > 0:
            ratios.append(len(pred) / len(ref))

    if not ratios:
        return 0.0, 0.0

    mean_ratio = statistics.mean(ratios)
    std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0.0

    return round(mean_ratio, 3), round(std_ratio, 3)
