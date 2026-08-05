"""
Cultural faithfulness evaluation for generated explanations.

Provides:
  1. A structured rubric judge (rule-based + optional LLM JSON judge)
  2. Aggregate Welch / mixed-model style group comparisons
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence

import numpy as np


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of cultural concept explanations.
Score the MODEL_OUTPUT against the GOLD_MEANING for the given concept.
Return ONLY valid JSON with these keys:
  meaning_correctness: float 0-1 (does it capture the gold meaning?)
  cultural_specificity: float 0-1 (is it culturally specific, not a vague paraphrase?)
  english_collapse: float 0-1 (1 = collapses to a single crude English word/gloss; 0 = preserves nuance)
  overall: float 0-1 (overall faithfulness)
  brief_rationale: short string
"""


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-zA-Z']+", (text or "").lower()))


def rule_based_scores(
    prediction: str,
    concept_word: str,
    language: str,
    meaning: str,
    translatability: str,
) -> Dict[str, float]:
    """
    Deterministic rubric proxy used when no LLM judge is available.

    Still richer than the old heuristic: meaning overlap, specificity
    (length + multi-word explanation), and English-collapse detection.
    """
    pred = (prediction or "").strip()
    meaning = meaning or ""
    pred_toks = _token_set(pred)
    meaning_toks = _token_set(meaning)

    # Meaning correctness via token overlap with gold meaning
    if meaning_toks:
        overlap = len(pred_toks & meaning_toks) / len(meaning_toks)
    else:
        overlap = 0.0
    meaning_correctness = float(min(1.0, overlap * 1.5))

    # Cultural specificity: mentions concept/language and is multi-clause
    specificity = 0.0
    if concept_word and concept_word.lower() in pred.lower():
        specificity += 0.25
    if language and language.lower() in pred.lower():
        specificity += 0.15
    n_words = len(pred.split())
    if n_words >= 12:
        specificity += 0.3
    if n_words >= 25:
        specificity += 0.15
    if any(p in pred for p in [".", ";", ","]):
        specificity += 0.15
    cultural_specificity = float(min(1.0, specificity))

    # English collapse: short single-gloss answers for untranslatables
    collapse = 0.0
    if n_words <= 4:
        collapse += 0.5
    if n_words <= 8:
        collapse += 0.2
    # Starts with "means X" with tiny X
    if re.match(r"^(means|is|refers to)\s+\w+\.?$", pred.lower()):
        collapse += 0.3
    if translatability == "translatable":
        # Collapse is less meaningful for controls; dampen
        collapse *= 0.5
    english_collapse = float(min(1.0, collapse))

    # Overall: reward meaning + specificity, penalize collapse
    overall = float(
        np.clip(
            0.45 * meaning_correctness
            + 0.40 * cultural_specificity
            - 0.25 * english_collapse
            + 0.20,
            0.0,
            1.0,
        )
    )

    return {
        "meaning_correctness": meaning_correctness,
        "cultural_specificity": cultural_specificity,
        "english_collapse": english_collapse,
        "overall": overall,
        "judge": "rule_based",
    }


def parse_judge_json(text: str) -> Optional[Dict]:
    """Extract JSON object from an LLM judge response."""
    if not text:
        return None
    text = text.strip()
    # Direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Fenced or embedded
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_judge_user_prompt(
    concept_word: str,
    language: str,
    meaning: str,
    model_output: str,
    translatability: str,
) -> str:
    return (
        f"CONCEPT: {concept_word}\n"
        f"LANGUAGE: {language}\n"
        f"TRANSLATABILITY: {translatability}\n"
        f"GOLD_MEANING: {meaning}\n"
        f"MODEL_OUTPUT:\n{model_output}\n"
    )


def score_generation(
    prediction: str,
    concept_word: str,
    language: str,
    meaning: str,
    translatability: str,
    llm_response: Optional[str] = None,
) -> Dict:
    """Combine optional LLM judge JSON with rule-based fallback."""
    base = rule_based_scores(
        prediction, concept_word, language, meaning, translatability
    )
    if llm_response:
        parsed = parse_judge_json(llm_response)
        if parsed:
            out = dict(base)
            for k in (
                "meaning_correctness",
                "cultural_specificity",
                "english_collapse",
                "overall",
            ):
                if k in parsed:
                    try:
                        out[k] = float(np.clip(float(parsed[k]), 0.0, 1.0))
                    except Exception:
                        pass
            out["judge"] = "llm"
            out["brief_rationale"] = parsed.get("brief_rationale", "")
            return out
    return base


def welch_group_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict:
    """Welch t-test + Cohen's d for two groups."""
    from scipy import stats

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return {"error": "insufficient_samples", "n_a": len(a), "n_b": len(b)}

    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-8
    d = (a.mean() - b.mean()) / pooled
    return {
        "test": "Welch's t-test",
        "t_statistic": float(t),
        "p_value": float(p),
        "cohens_d": float(d),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_a": float(a.std(ddof=1)),
        "std_b": float(b.std(ddof=1)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "significant_005": bool(p < 0.05),
    }


def aggregate_faithfulness(records: List[Dict], score_key: str = "overall") -> Dict:
    """
    Aggregate scored generations.

    Each record needs: translatability, template_name, language, scores dict.
    """
    untrans = [
        r["scores"][score_key]
        for r in records
        if r.get("translatability") == "untranslatable" and score_key in r.get("scores", {})
    ]
    trans = [
        r["scores"][score_key]
        for r in records
        if r.get("translatability") == "translatable" and score_key in r.get("scores", {})
    ]

    by_template = {}
    for tmpl in sorted({r.get("template_name") for r in records}):
        ua = [
            r["scores"][score_key]
            for r in records
            if r.get("template_name") == tmpl
            and r.get("translatability") == "untranslatable"
            and score_key in r.get("scores", {})
        ]
        ta = [
            r["scores"][score_key]
            for r in records
            if r.get("template_name") == tmpl
            and r.get("translatability") == "translatable"
            and score_key in r.get("scores", {})
        ]
        by_template[tmpl] = welch_group_test(ua, ta)

    by_metric = {}
    for metric in (
        "meaning_correctness",
        "cultural_specificity",
        "english_collapse",
        "overall",
    ):
        ua = [
            r["scores"][metric]
            for r in records
            if r.get("translatability") == "untranslatable" and metric in r.get("scores", {})
        ]
        ta = [
            r["scores"][metric]
            for r in records
            if r.get("translatability") == "translatable" and metric in r.get("scores", {})
        ]
        by_metric[metric] = welch_group_test(ua, ta)

    return {
        "n_records": len(records),
        "overall_untrans_vs_trans": welch_group_test(untrans, trans),
        "by_template": by_template,
        "by_metric": by_metric,
    }
