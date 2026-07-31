"""
Step 2: Stratified analysis of per-prompt English probability.

The pooled comparison mixes 10 languages and 3 prompt templates, each with a
different baseline next-token distribution. This module stratifies the
analysis by template and by language, and fits a mixed-effects model with
template and language as fixed effects.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

from src.analysis.english_probability import welch_ttest


def stratify_by_template(
    p_eng: np.ndarray,
    metadata: List[Dict],
    layer_indices: List[int],
    mid_layer_indices: List[int],
) -> Dict:
    """Per-template group comparison of mid-layer per-prompt English probability."""
    templates = sorted({m["template_name"] for m in metadata})
    out = {}
    mid_rows = [layer_indices.index(l) for l in mid_layer_indices]
    for tpl in templates:
        idx = [i for i, m in enumerate(metadata) if m["template_name"] == tpl]
        mask = np.array([metadata[i]["translatability"] == "untranslatable" for i in idx])
        mid_mean = p_eng[mid_rows][:, idx].mean(axis=0)
        out[tpl] = {
            "n_untrans": int(mask.sum()),
            "n_trans": int((~mask).sum()),
            "welch_midlayer": welch_ttest(mid_mean[mask], mid_mean[~mask]),
            "per_layer_diff": {
                str(layer_indices[i]): float(
                    p_eng[i][idx][mask].mean() - p_eng[i][idx][~mask].mean()
                )
                for i in range(len(layer_indices))
            },
        }
    return out


def stratify_by_language(
    p_eng: np.ndarray,
    metadata: List[Dict],
    layer_indices: List[int],
    mid_layer_indices: List[int],
) -> Dict:
    """Per-language group comparison of mid-layer per-prompt English probability."""
    languages = sorted({m["language_code"] for m in metadata})
    mid_rows = [layer_indices.index(l) for l in mid_layer_indices]
    out = {}
    for lang in languages:
        idx = [i for i, m in enumerate(metadata) if m["language_code"] == lang]
        mask = np.array([metadata[i]["translatability"] == "untranslatable" for i in idx])
        if mask.sum() < 3 or (~mask).sum() < 3:
            continue
        mid_mean = p_eng[mid_rows][:, idx].mean(axis=0)
        out[lang] = {
            "n_untrans": int(mask.sum()),
            "n_trans": int((~mask).sum()),
            "welch_midlayer": welch_ttest(mid_mean[mask], mid_mean[~mask]),
        }
    return out


def mixed_model_with_strata(
    p_eng: np.ndarray,
    metadata: List[Dict],
    layer_indices: List[int],
) -> Dict:
    """
    logit(p_eng) ~ untrans * depth + template + language, random intercept per
    prompt. Returns coefficients and p-values for translatability terms.
    """
    import pandas as pd
    from statsmodels.formula.api import mixedlm

    n_prompts = p_eng.shape[1]
    depth = np.linspace(0.0, 1.0, len(layer_indices))
    rows = []
    for li in range(len(layer_indices)):
        for i in range(n_prompts):
            p = min(max(p_eng[li, i], 1e-6), 1.0 - 1e-6)
            rows.append({
                "logit_p": float(np.log(p / (1.0 - p))),
                "untrans": int(metadata[i]["translatability"] == "untranslatable"),
                "depth": float(depth[li]),
                "template": metadata[i]["template_name"],
                "language": metadata[i]["language_code"],
                "prompt": i,
            })
    df = pd.DataFrame(rows)
    formula = "logit_p ~ untrans * depth + C(template) + C(language)"
    try:
        fit = mixedlm(formula, df, groups=df["prompt"]).fit(reml=False, maxiter=500)
        out = {}
        for name in ["untrans", "untrans:depth"]:
            try:
                out[name] = {
                    "coef": float(fit.params[name]),
                    "p_value": float(fit.pvalues[name]),
                }
            except Exception:
                pass
        return out
    except Exception as e:
        return {"error": str(e)}


def run_stratified_analysis(
    activations_dir: str,
    output_dir: str,
    model_name: str,
    position: str = "last",
    device: Optional[str] = None,
) -> Dict:
    """Full stratified analysis for one model.

    position: "last" (final-token hidden states) or "concept" (concept-word
    token hidden states).
    """
    from src.utils.io import (
        load_activations,
        load_concept_activations,
        load_lm_head_from_file,
        load_final_norm_from_file,
        load_metadata,
    )
    from transformers import AutoTokenizer

    from src.analysis.english_probability import (
        compute_english_probability_mass,
        mid_layer_indices,
    )

    print("  Loading activations...")
    if position == "concept":
        hidden_states = load_concept_activations(activations_dir)
    else:
        hidden_states = load_activations(activations_dir)
    metadata = load_metadata(activations_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lm_head_weights, lm_head_bias = load_lm_head_from_file(
        os.path.join(activations_dir, "lm_head.npz")
    )
    final_norm_gamma = load_final_norm_from_file(
        os.path.join(activations_dir, "final_norm.npz")
    )

    print("  Computing per-prompt English probability mass...")
    p_eng, layer_indices = compute_english_probability_mass(
        hidden_states, tokenizer, lm_head_weights, lm_head_bias, final_norm_gamma,
        device=device,
    )
    mid_layers = mid_layer_indices(layer_indices)

    print("  Stratifying by template...")
    by_template = stratify_by_template(p_eng, metadata, layer_indices, mid_layers)

    print("  Stratifying by language...")
    by_language = stratify_by_language(p_eng, metadata, layer_indices, mid_layers)

    print("  Mixed-effects model with template + language fixed effects...")
    mixed = mixed_model_with_strata(p_eng, metadata, layer_indices)

    return {
        "by_template": by_template,
        "by_language": by_language,
        "mixed_model": mixed,
        "mid_layer_indices": mid_layers,
    }
