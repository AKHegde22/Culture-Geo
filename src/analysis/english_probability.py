"""
Per-prompt graded English probability analysis.

The group-level English Routing Rate (ERR) collapses each layer into a single
fraction (top-1 is English), destroying per-prompt information and forcing
statistical tests to use layers as units (n=32). This module instead computes,
for every prompt and every layer, the softmax probability mass assigned to
English tokens:

    p_eng(i, l) = sum_{v in English} softmax(logit_v(l))

The result is a continuous per-prompt outcome that permits properly powered
tests (n = number of prompts) and a mixed-effects model with prompt as a
random effect.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.analysis.logit_lens import apply_final_norm, _get_english_token_ids


def compute_english_probability_mass(
    hidden_states: Dict[int, np.ndarray],
    tokenizer,
    lm_head_weights: np.ndarray,
    lm_head_bias: Optional[np.ndarray] = None,
    final_norm_gamma: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> tuple:
    """
    Compute per-prompt English probability mass at every layer.

    Args:
        device: If "cuda", uses a torch GPU fast path that mirrors the numpy
            math exactly (float32, RMSNorm, stable softmax). Otherwise uses
            the pure-numpy implementation.

    Returns:
        p_eng: np.ndarray [n_layers, n_prompts] of English probability mass
        layer_indices: list of layer indices aligned with p_eng rows
    """
    lm_head_weights = lm_head_weights.astype(np.float32)
    if lm_head_bias is not None:
        lm_head_bias = lm_head_bias.astype(np.float32)

    english_ids = np.array(sorted(_get_english_token_ids(tokenizer)), dtype=np.int64)

    layer_indices = sorted(hidden_states.keys())
    n_prompts = next(iter(hidden_states.values())).shape[0]
    p_eng = np.zeros((len(layer_indices), n_prompts), dtype=np.float32)

    if device == "cuda":
        return _compute_english_probability_mass_torch(
            hidden_states, english_ids, lm_head_weights, lm_head_bias,
            final_norm_gamma, layer_indices, n_prompts, p_eng,
        )

    for li, layer_idx in enumerate(layer_indices):
        hs = hidden_states[layer_idx].astype(np.float32)
        layer_logits = apply_final_norm(hs, final_norm_gamma) @ lm_head_weights
        if lm_head_bias is not None:
            layer_logits += lm_head_bias

        # Numerically stable softmax; gather English mass
        logit_max = layer_logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(layer_logits - logit_max)
        denom = exp_logits.sum(axis=-1)
        numer = exp_logits[:, english_ids].sum(axis=-1)
        p_eng[li] = numer / denom

    return p_eng, layer_indices


def _compute_english_probability_mass_torch(
    hidden_states: Dict[int, np.ndarray],
    english_ids: np.ndarray,
    lm_head_weights: np.ndarray,
    lm_head_bias: Optional[np.ndarray],
    final_norm_gamma: Optional[np.ndarray],
    layer_indices: List[int],
    n_prompts: int,
    p_eng: np.ndarray,
) -> tuple:
    """GPU (torch) fast path mirroring the numpy implementation."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    W = torch.from_numpy(lm_head_weights).cuda()
    b = torch.from_numpy(lm_head_bias).cuda() if lm_head_bias is not None else None
    g = torch.from_numpy(final_norm_gamma.astype(np.float32)).cuda() \
        if final_norm_gamma is not None else None
    eng_ids = torch.from_numpy(english_ids).cuda()

    for li, layer_idx in enumerate(layer_indices):
        hs = torch.from_numpy(hidden_states[layer_idx].astype(np.float32)).cuda()
        if g is not None:
            rms = torch.sqrt(torch.mean(hs * hs, dim=-1, keepdim=True) + 1e-6)
            hs = hs / rms * g
        logits = hs @ W
        if b is not None:
            logits = logits + b
        logit_max = logits.max(dim=-1, keepdim=True).values
        exp_logits = (logits - logit_max).exp()
        denom = exp_logits.sum(dim=-1)
        numer = exp_logits.index_select(1, eng_ids).sum(dim=-1)
        p_eng[li] = (numer / denom).float().cpu().numpy()

    return p_eng, layer_indices


def mid_layer_indices(layer_indices: List[int], low: float = 0.25, high: float = 0.75) -> List[int]:
    """Layers in the middle depth band (excludes early artifacts and the output)."""
    n = len(layer_indices)
    lo = int(round(low * n))
    hi = int(round(high * n))
    return layer_indices[lo:max(hi, lo + 1)]


def welch_ttest(group_a: np.ndarray, group_b: np.ndarray) -> Dict:
    """Welch's t-test on two vectors of per-prompt values."""
    from scipy import stats

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    pooled_std = np.sqrt((group_a.std() ** 2 + group_b.std() ** 2) / 2)
    cohens_d = (group_a.mean() - group_b.mean()) / max(pooled_std, 1e-8)
    se = np.sqrt(group_a.var() / len(group_a) + group_b.var() / len(group_b))
    se_d = se / max(pooled_std, 1e-8)
    return {
        "test": "Welch's t-test (per-prompt)",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_95_lower": float(cohens_d - 1.96 * se_d),
        "ci_95_upper": float(cohens_d + 1.96 * se_d),
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "n_a": int(len(group_a)),
        "n_b": int(len(group_b)),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
    }


def permutation_test_by_layer(
    p_eng: np.ndarray,
    group_mask: np.ndarray,
    layer_indices: List[int],
    n_permutations: int = 5000,
    seed: int = 0,
) -> Dict[int, float]:
    """
    Per-layer permutation test: shuffle prompt group labels, recompute the
    absolute mean difference between groups, and return empirical p-values.
    """
    rng = np.random.default_rng(seed)
    n_prompts = p_eng.shape[1]
    observed = np.abs(
        p_eng[:, group_mask].mean(axis=1) - p_eng[:, ~group_mask].mean(axis=1)
    )
    count = np.zeros(p_eng.shape[0], dtype=int)
    labels = group_mask.copy()
    for _ in range(n_permutations):
        rng.shuffle(labels)
        perm = np.abs(
            p_eng[:, labels].mean(axis=1) - p_eng[:, ~labels].mean(axis=1)
        )
        count += (perm >= observed).astype(int)
    pvals = (count + 1) / (n_permutations + 1)
    return {layer_indices[i]: float(pvals[i]) for i in range(len(layer_indices))}


def mixed_effects_logit(
    p_eng: np.ndarray,
    group_mask: np.ndarray,
    layer_indices: List[int],
) -> Dict:
    """
    Mixed-effects model on logit(p_eng) ~ translatability * normalized_depth,
    with random intercept per prompt. Returns coefficient estimates and p-values.
    """
    import pandas as pd
    from statsmodels.formula.api import mixedlm

    n_prompts = p_eng.shape[1]
    depth = np.linspace(0.0, 1.0, len(layer_indices))
    rows = []
    for li, layer_idx in enumerate(layer_indices):
        for i in range(n_prompts):
            p = min(max(p_eng[li, i], 1e-6), 1.0 - 1e-6)
            rows.append({
                "logit_p": float(np.log(p / (1.0 - p))),
                "untrans": int(bool(group_mask[i])),
                "depth": float(depth[li]),
                "prompt": i,
            })
    df = pd.DataFrame(rows)
    try:
        model = mixedlm("logit_p ~ untrans * depth", df, groups=df["prompt"])
        fit = model.fit(reml=False, maxiter=500)
        out = {}
        for name in ["Intercept", "untrans", "depth", "untrans:depth"]:
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


def run_english_probability_analysis(
    activations_dir: str,
    output_dir: str,
    model_name: str,
    device: Optional[str] = None,
) -> Dict:
    """Full per-prompt English probability analysis for one model."""
    from src.utils.io import (
        load_activations,
        load_lm_head_from_file,
        load_final_norm_from_file,
        load_metadata,
    )
    from transformers import AutoTokenizer

    print("  Loading activations...")
    hidden_states = load_activations(activations_dir)
    metadata = load_metadata(activations_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lm_head_weights, lm_head_bias = load_lm_head_from_file(
        os.path.join(activations_dir, "lm_head.npz")
    )
    final_norm_gamma = load_final_norm_from_file(
        os.path.join(activations_dir, "final_norm.npz")
    )
    if lm_head_weights is None:
        raise RuntimeError("LM head weights required for English probability analysis")

    print("  Computing per-prompt English probability mass...")
    p_eng, layer_indices = compute_english_probability_mass(
        hidden_states, tokenizer, lm_head_weights, lm_head_bias, final_norm_gamma,
        device=device,
    )

    group_mask = np.array(
        [m["translatability"] == "untranslatable" for m in metadata], dtype=bool
    )

    mid_layers = mid_layer_indices(layer_indices)
    mid_rows = [layer_indices.index(l) for l in mid_layers]
    per_prompt_mid = p_eng[mid_rows].mean(axis=0)

    results = {
        "n_prompts": int(p_eng.shape[1]),
        "n_layers": len(layer_indices),
        "mid_layer_indices": mid_layers,
        "english_vocab_size": int(np.array(sorted(_get_english_token_ids(tokenizer))).size),
        "group_n": {
            "untranslatable": int(group_mask.sum()),
            "translatable": int((~group_mask).sum()),
        },
        "per_layer": {
            "untranslatable_mean": {
                str(l): float(p_eng[i][group_mask].mean())
                for i, l in enumerate(layer_indices)
            },
            "translatable_mean": {
                str(l): float(p_eng[i][~group_mask].mean())
                for i, l in enumerate(layer_indices)
            },
        },
        "per_prompt_midlayer": {
            "untranslatable": per_prompt_mid[group_mask].tolist(),
            "translatable": per_prompt_mid[~group_mask].tolist(),
        },
    }

    print("  Saving per-prompt probability matrix...")
    np.savez_compressed(
        os.path.join(output_dir, "p_eng.npz"),
        p_eng=p_eng,
        layer_indices=np.array(layer_indices),
        group_mask=group_mask,
    )

    print("  Permutation tests per layer...")
    results["permutation_p_by_layer"] = permutation_test_by_layer(
        p_eng, group_mask, layer_indices
    )

    print("  Welch test on per-prompt mid-layer English probability...")
    results["welch_midlayer"] = welch_ttest(
        per_prompt_mid[group_mask], per_prompt_mid[~group_mask]
    )

    print("  Mixed-effects model (logit p_eng ~ untrans * depth)...")
    results["mixed_model"] = mixed_effects_logit(p_eng, group_mask, layer_indices)

    return results
