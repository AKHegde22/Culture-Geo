"""
Evaluation metrics for generation quality.

Path B uses the structured cultural-faithfulness rubric in
`src.analysis.faithfulness`. This module re-exports those helpers and keeps
ROUGE-L as a secondary lexical overlap metric against gold meanings.
"""

from typing import Dict, List, Optional

import numpy as np

from src.analysis.faithfulness import (
    JUDGE_SYSTEM_PROMPT,
    aggregate_faithfulness,
    build_judge_user_prompt,
    rule_based_scores,
    score_generation,
    welch_group_test,
)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
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
    translatability: str = "untranslatable",
) -> float:
    """Return overall faithfulness score from the Path-B rubric."""
    return score_generation(
        prediction, concept_word, language, meaning, translatability
    )["overall"]


def compute_translation_quality(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Aggregate ROUGE-L against gold meanings (secondary metric)."""
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        rouge_scores.append(compute_rouge_l(pred, ref))

    return {
        "rouge_l_mean": float(np.mean(rouge_scores)) if rouge_scores else 0.0,
        "rouge_l_std": float(np.std(rouge_scores)) if rouge_scores else 0.0,
        "rouge_l_median": float(np.median(rouge_scores)) if rouge_scores else 0.0,
        "n_samples": len(predictions),
    }


__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "aggregate_faithfulness",
    "build_judge_user_prompt",
    "compute_cultural_faithfulness",
    "compute_rouge_l",
    "compute_translation_quality",
    "rule_based_scores",
    "score_generation",
    "welch_group_test",
]
