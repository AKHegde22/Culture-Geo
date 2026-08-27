"""
Cultural faithfulness evaluation for generated explanations.

Provides:
  1. A structured, condition-blind rubric judge (rule-based + optional blinded LLM JSON judge)
     Three fixed dimensions:
       - semantic_correctness: factual accuracy, absence of hallucination/degenerative loops
       - nuance_retained: depth of explanation, avoids crude single-word collapse
       - cultural_adequacy: grounds the concept in its cultural community, context, or usage
       - overall: unweighted mean of the three dimensions
  2. Statistical evaluation (Welch's t-test with 95% CIs, paired permutation tests, confound regression)
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of cultural concept explanations.
Score the MODEL_OUTPUT for the given concept and language against the REFERENCE_MEANING.
Evaluate blindly without regard to whether the concept has a single-word English equivalent.
Return ONLY valid JSON with these keys:
  semantic_correctness: float 0-1 (factual accuracy, absence of hallucination/degenerative repetition, coherent definition)
  nuance_retained: float 0-1 (depth of explanation, avoids crude single-word collapse, conveys subtleties and connotations)
  cultural_adequacy: float 0-1 (grounds the concept in its cultural community, social context, traditions, or usage)
  overall: float 0-1 (unweighted average of the three dimensions above)
  brief_rationale: short string explaining the score
"""


def _token_set(text: str) -> set:
    return set(re.findall(r"[a-zA-Z']+", (text or "").lower()))


def rule_based_scores(
    prediction: str,
    concept_word: str,
    language: str,
    meaning: str,
    translatability: Optional[str] = None,  # Kept for signature compatibility; STRICTLY UNUSED for blinding
) -> Dict[str, float]:
    """
    Deterministic, condition-blind rubric scoring three fixed dimensions:
      1. semantic_correctness: Checks absence of degenerative repetition, definitional framing,
         and conceptual substance (without raw string overlap penalizing nuanced concepts).
      2. nuance_retained: Evaluates multi-clause depth and nuance markers while penalizing
         single-word copula collapses identically for ALL concepts.
      3. cultural_adequacy: Evaluates explicit grounding in source language/culture and contextual usage.
    """
    pred = (prediction or "").strip()
    words = pred.split()
    n_words = len(words)

    if n_words < 3:
        return {
            "semantic_correctness": 0.0,
            "nuance_retained": 0.0,
            "cultural_adequacy": 0.0,
            "overall": 0.0,
            "judge": "blinded_rule_based",
        }

    # 1. Semantic correctness
    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / max(n_words, 1)

    # Repetition penalty for degenerative loops
    rep_penalty = 0.0
    if n_words >= 20 and ttr < 0.45:
        rep_penalty = min(0.5, (0.45 - ttr) * 2.0)

    # Definitional framing
    has_def_framing = bool(
        re.search(
            r"\b(means|refers to|concept of|feeling of|practice of|state of|habit of|describes|used to describe|tradition of)\b",
            pred.lower(),
        )
    )

    # Semantic content alignment
    meaning_tokens = set(re.findall(r"[a-zA-Z]{3,}", (meaning or "").lower()))
    stopwords = {"the", "and", "for", "that", "with", "from", "something", "someone", "this"}
    content_meaning = meaning_tokens - stopwords

    if content_meaning:
        pred_content = set(re.findall(r"[a-zA-Z]{3,}", pred.lower()))
        overlap_count = len(pred_content & content_meaning)
        sem_match = min(1.0, overlap_count / max(min(len(content_meaning), 4), 1))
    else:
        sem_match = 0.5

    semantic_correctness = float(
        np.clip(
            0.35 * sem_match
            + 0.35 * (1.0 if has_def_framing else 0.5)
            + 0.30 * min(1.0, ttr * 1.5)
            - rep_penalty,
            0.0,
            1.0,
        )
    )

    # 2. Nuance retained (0-1)
    # Collapse penalty is label-invariant: applies identically to translatable and untranslatable
    collapse = 0.0
    if n_words <= 5:
        collapse = 0.6
    elif n_words <= 9:
        collapse = 0.3
    if re.match(r"^(means|is|refers to)\s+[\"\'\w\s]{1,15}\.?$", pred.lower()):
        collapse = max(collapse, 0.5)

    nuance_markers = bool(
        re.search(
            r"\b(more than|specifically|nuance|connotation|cultural|significance|lifestyle|philosophy|associated with|deep|beyond|implies)\b",
            pred.lower(),
        )
    )
    has_clauses = bool(re.search(r"[,;:\(\)]", pred))
    length_score = min(1.0, n_words / 25.0)

    nuance_retained = float(
        np.clip(
            0.40 * length_score
            + 0.30 * (1.0 if has_clauses else 0.4)
            + 0.30 * (1.0 if nuance_markers else 0.5)
            - collapse,
            0.0,
            1.0,
        )
    )

    # 3. Cultural / contextual adequacy (0-1)
    mentions_lang_or_concept = 0.0
    if concept_word and concept_word.lower() in pred.lower():
        mentions_lang_or_concept += 0.35
    if language and language.lower() in pred.lower():
        mentions_lang_or_concept += 0.25

    contextual_usage = bool(
        re.search(
            r"\b(culture|tradition|custom|people|context|society|community|everyday|practice|life|situation|used when)\b",
            pred.lower(),
        )
    )
    cultural_adequacy = float(
        np.clip(
            mentions_lang_or_concept + 0.40 * (1.0 if contextual_usage else 0.3),
            0.0,
            1.0,
        )
    )

    overall = float((semantic_correctness + nuance_retained + cultural_adequacy) / 3.0)

    return {
        "semantic_correctness": semantic_correctness,
        "nuance_retained": nuance_retained,
        "cultural_adequacy": cultural_adequacy,
        "overall": overall,
        # Backward compatibility aliases
        "meaning_correctness": semantic_correctness,
        "cultural_specificity": cultural_adequacy,
        "english_collapse": collapse,
        "judge": "blinded_rule_based",
    }


def parse_judge_json(text: str) -> Optional[Dict]:
    """Extract JSON object from an LLM judge response."""
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
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
    translatability: Optional[str] = None,  # Ignored to ensure blinded evaluation
) -> str:
    return (
        f"CONCEPT: {concept_word}\n"
        f"LANGUAGE: {language}\n"
        f"REFERENCE_MEANING: {meaning}\n"
        f"MODEL_OUTPUT:\n{model_output}\n"
    )


def score_generation(
    prediction: str,
    concept_word: str,
    language: str,
    meaning: str,
    translatability: Optional[str] = None,
    llm_response: Optional[str] = None,
) -> Dict:
    """Combine optional blinded LLM judge JSON with rule-based fallback."""
    base = rule_based_scores(
        prediction, concept_word, language, meaning, translatability
    )
    if llm_response:
        parsed = parse_judge_json(llm_response)
        if parsed:
            out = dict(base)
            for k in (
                "semantic_correctness",
                "nuance_retained",
                "cultural_adequacy",
                "overall",
                "meaning_correctness",
                "cultural_specificity",
            ):
                if k in parsed:
                    try:
                        out[k] = float(np.clip(float(parsed[k]), 0.0, 1.0))
                    except Exception:
                        pass
            out["judge"] = "blinded_llm"
            out["brief_rationale"] = parsed.get("brief_rationale", "")
            return out
    return base


def welch_group_test(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict:
    """Welch t-test + Cohen's d with analytical 95% confidence interval."""
    from scipy import stats

    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return {"error": "insufficient_samples", "n_a": len(a), "n_b": len(b)}

    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-8
    d = (a.mean() - b.mean()) / pooled

    # 95% CI for Cohen's d
    se_d = np.sqrt((len(a) + len(b)) / (len(a) * len(b)) + (d ** 2) / (2 * (len(a) + len(b))))
    ci_low = float(d - 1.96 * se_d)
    ci_high = float(d + 1.96 * se_d)

    return {
        "test": "Welch's t-test",
        "t_statistic": float(t),
        "p_value": float(p),
        "cohens_d": float(d),
        "ci_95": [ci_low, ci_high],
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_a": float(a.std(ddof=1)),
        "std_b": float(b.std(ddof=1)),
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "significant_005": bool(p < 0.05),
    }


def paired_permutation_test(
    scores_u: Sequence[float],
    scores_t: Sequence[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Exact Monte Carlo sign-flip permutation test for one-to-one matched pairs.
    """
    u = np.asarray(scores_u, dtype=np.float64)
    t = np.asarray(scores_t, dtype=np.float64)
    assert len(u) == len(t), "Paired permutation test requires equal length sequences"

    diffs = u - t
    actual_mean = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, len(diffs)))
    perm_means = (signs * diffs).mean(axis=1)

    p_value = float((np.abs(perm_means) >= np.abs(actual_mean)).mean())
    pooled_std = float(np.sqrt((u.var(ddof=1) + t.var(ddof=1)) / 2) + 1e-8)
    cohens_d = float(actual_mean / pooled_std)

    se_d = np.sqrt(1 / len(u) + (cohens_d ** 2) / (2 * len(u)))
    ci_low = float(cohens_d - 1.96 * se_d)
    ci_high = float(cohens_d + 1.96 * se_d)

    return {
        "test": "Paired Permutation Test",
        "n_pairs": int(len(u)),
        "mean_diff": actual_mean,
        "cohens_d": cohens_d,
        "ci_95": [ci_low, ci_high],
        "p_value": p_value,
        "significant_005": bool(p_value < 0.05),
    }


def confound_regression(records: List[Dict], score_key: str = "overall") -> Dict:
    """
    Ordinary Least Squares regression controlling for word length, token count,
    language, and category to assess whether the translatability effect remains.
    """
    import statsmodels.api as sm
    import pandas as pd

    rows = []
    for r in records:
        sc = r.get("scores", {}).get(score_key)
        if sc is None:
            continue
        word = r.get("concept_word", "")
        rows.append({
            "score": sc,
            "is_untranslatable": 1.0 if r.get("translatability") == "untranslatable" else 0.0,
            "char_len": float(len(word)),
            "token_count": float(len(word.split())),
            "language": r.get("language", "unknown"),
            "category": r.get("category", "unknown"),
            "template": r.get("template_name", "unknown"),
        })

    df = pd.DataFrame(rows)
    if len(df) < 10:
        return {"error": "insufficient_data"}

    # Dummy encode language and category
    df_reg = pd.get_dummies(df, columns=["language", "category", "template"], drop_first=True, dtype=float)
    y = df_reg["score"]
    X = df_reg.drop(columns=["score"])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    coef = float(model.params.get("is_untranslatable", 0.0))
    pval = float(model.pvalues.get("is_untranslatable", 1.0))
    ci = model.conf_int().loc["is_untranslatable"].tolist()

    return {
        "is_untranslatable_coef": coef,
        "is_untranslatable_pval": pval,
        "ci_95": [float(ci[0]), float(ci[1])],
        "r_squared": float(model.rsquared),
        "n_obs": int(model.nobs),
        "significant_005": bool(pval < 0.05),
    }


def aggregate_faithfulness(records: List[Dict], score_key: str = "overall") -> Dict:
    """
    Aggregate scored generations with Welch, permutation, and regression tests.
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
        "semantic_correctness",
        "nuance_retained",
        "cultural_adequacy",
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

    reg = confound_regression(records, score_key=score_key)

    return {
        "n_records": len(records),
        "overall_untrans_vs_trans": welch_group_test(untrans, trans),
        "by_template": by_template,
        "by_metric": by_metric,
        "confound_regression": reg,
    }
