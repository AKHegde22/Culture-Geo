"""
Evaluation metrics for generation quality.
"""

from typing import Dict, List, Optional

import numpy as np


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between prediction and reference.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        # Fallback: simple token overlap
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not ref_tokens:
            return 0.0
        common = pred_tokens & ref_tokens
        precision = len(common) / max(len(pred_tokens), 1)
        recall = len(common) / max(len(ref_tokens), 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def compute_cultural_faithfulness(
    prediction: str,
    concept_word: str,
    language: str,
    meaning: str,
) -> float:
    """
    Simple heuristic for cultural faithfulness.

    Checks if the prediction:
    1. Mentions the concept word
    2. References the source language
    3. Captures key aspects of the meaning

    This is a simplified proxy - for the paper, human evaluation is preferred.
    """
    score = 0.0

    # Check if concept word is mentioned
    if concept_word.lower() in prediction.lower():
        score += 0.3

    # Check if source language is mentioned
    if language.lower() in prediction.lower():
        score += 0.2

    # Check key meaning words
    if meaning:
        meaning_words = set(meaning.lower().split())
        pred_words = set(prediction.lower().split())
        overlap = meaning_words & pred_words
        score += min(0.5, len(overlap) / max(len(meaning_words), 1) * 0.5)

    return min(1.0, score)


def compute_translation_quality(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute translation quality metrics across a set of predictions.

    Args:
        predictions: List of model-generated explanations
        references: List of reference translations/explanations

    Returns:
        Dictionary of aggregate metrics
    """
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        rouge_scores.append(compute_rouge_l(pred, ref))

    return {
        "rouge_l_mean": float(np.mean(rouge_scores)),
        "rouge_l_std": float(np.std(rouge_scores)),
        "rouge_l_median": float(np.median(rouge_scores)),
        "n_samples": len(predictions),
    }
