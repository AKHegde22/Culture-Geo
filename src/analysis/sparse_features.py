"""
Step 4: Sparse-feature analysis of hidden activations.

Aggregate residual-stream statistics may miss differences that live in sparse,
feature-level structure (as in sparse autoencoder analyses). This module learns
a sparse dictionary of features (via sklearn DictionaryLearning) from the
concept-token hidden states and tests whether the sparse code distributions
differ between untranslatable and translatable concepts.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np


def learn_dictionary(
    X: np.ndarray,
    n_components: int = 512,
    alpha: float = 1.0,
    positive_code: bool = False,
    max_iter: int = 200,
    seed: int = 0,
):
    """
    Learn a sparse dictionary D and codes C such that X ~ C @ D.

    Args:
        X: [n_samples, d_model] activations (should be L2-normalized per sample)

    Returns:
        D: [n_components, d_model] dictionary atoms
        C: [n_samples, n_components] sparse codes
    """
    from sklearn.decomposition import DictionaryLearning

    X = X.astype(np.float64)
    norms = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8
    Xn = X / norms

    model = DictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        positive_code=positive_code,
        max_iter=max_iter,
        random_state=seed,
        transform_algorithm="lasso_lars",
        fit_algorithm="lars",
    )
    C = model.fit_transform(Xn)
    D = model.components_
    return D, C


def per_feature_group_stats(C: np.ndarray, group_mask: np.ndarray) -> Dict:
    """Per-feature mean activation and fraction-active per group."""
    n_features = C.shape[1]
    out = {
        "n_features": int(n_features),
        "n_untrans": int(group_mask.sum()),
        "n_trans": int((~group_mask).sum()),
    }
    mean_u = C[group_mask].mean(axis=0)
    mean_t = C[~group_mask].mean(axis=0)
    active_u = (C[group_mask] > 0).mean(axis=0)
    active_t = (C[~group_mask] > 0).mean(axis=0)

    # Cohen's d per feature (untrans vs trans)
    var_u = C[group_mask].var(axis=0, ddof=1)
    var_t = C[~group_mask].var(axis=0, ddof=1)
    pooled = np.sqrt((var_u + var_t) / 2) + 1e-8
    d = (mean_u - mean_t) / pooled

    out["mean_untrans"] = mean_u.tolist()
    out["mean_trans"] = mean_t.tolist()
    out["active_untrans"] = active_u.tolist()
    out["active_trans"] = active_t.tolist()
    out["cohens_d"] = d.tolist()
    out["n_features_d_abs_gt_0_5"] = int((np.abs(d) > 0.5).sum())
    out["n_features_d_abs_gt_1_0"] = int((np.abs(d) > 1.0).sum())
    return out


def classify_translatability(
    C: np.ndarray,
    metadata: List[Dict],
    n_folds: int = 5,
    seed: int = 0,
) -> Dict:
    """
    Can sparse codes predict translatability? Language-stratified logistic
    regression with 5-fold CV, reported as balanced accuracy.

    Also returns a permutation p-value (shuffled labels).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score

    y = np.array(
        [1 if m["translatability"] == "untranslatable" else 0 for m in metadata]
    )
    languages = np.array([m["language_code"] for m in metadata])

    # Only use samples from languages that contain both groups
    keep = np.zeros(len(y), dtype=bool)
    for lang in np.unique(languages):
        mask = languages == lang
        if y[mask].sum() > 0 and (1 - y[mask]).sum() > 0:
            keep |= mask
    if keep.sum() < 20:
        return {"error": "Not enough language-balanced samples"}

    Ck, yk = C[keep], y[keep]

    # Stratified CV within language: split each language separately
    def stratified_cv_accuracy(Cx, yx, rng_seed):
        rng = np.random.default_rng(rng_seed)
        scores = []
        for lang in np.unique(languages[keep]):
            idx = np.where(languages[keep] == lang)[0]
            if len(idx) < 4:
                continue
            skf = StratifiedKFold(
                n_splits=min(n_folds, int(yx[idx].sum()), int((1 - yx[idx]).sum())),
                shuffle=True, random_state=rng_seed,
            )
            for tr, te in skf.split(idx, yx[idx]):
                clf = LogisticRegression(C=1.0, max_iter=2000)
                clf.fit(Cx[idx][tr], yx[idx][tr])
                scores.append(balanced_accuracy_score(yx[idx][te], clf.predict(Cx[idx][te])))
        return float(np.mean(scores)) if scores else 0.5

    observed = stratified_cv_accuracy(Ck, yk, seed)

    # Permutation test (shuffle labels within language)
    rng = np.random.default_rng(seed)
    perm_scores = []
    for _ in range(100):
        perm_y = yk.copy()
        for lang in np.unique(languages[keep]):
            idx = np.where(languages[keep] == lang)[0]
            rng.shuffle(perm_y[idx])
        perm_scores.append(stratified_cv_accuracy(Ck, perm_y, seed))
    perm_scores = np.array(perm_scores)
    p_value = float((perm_scores >= observed).mean() + 1e-3)

    return {
        "balanced_accuracy": observed,
        "permutation_p": min(p_value, 1.0),
        "n_samples_kept": int(keep.sum()),
        "n_permutations": 100,
    }


def run_sparse_analysis(
    hidden_states: Dict[int, np.ndarray],
    metadata: List[Dict],
    layer_idx: int,
    n_components: int = 512,
    alpha: float = 1.0,
    seed: int = 0,
) -> Dict:
    """Full sparse-feature analysis at a single layer."""
    group_mask = np.array(
        [m["translatability"] == "untranslatable" for m in metadata], dtype=bool
    )
    X = hidden_states[layer_idx].astype(np.float32)

    print(f"    Learning {n_components} sparse features at layer {layer_idx}...")
    D, C = learn_dictionary(
        X, n_components=n_components, alpha=alpha, seed=seed
    )

    results = {
        "layer_idx": layer_idx,
        "n_components": int(n_components),
        "alpha": float(alpha),
        "feature_stats": per_feature_group_stats(C, group_mask),
        "classify": classify_translatability(C, metadata, seed=seed),
        "reconstruction": {
            "explained_variance": float(
                1.0 - np.mean((X - C @ D) ** 2) / np.mean(X**2)
            )
        },
    }
    return results
