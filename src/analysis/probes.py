"""
Linear probes for language identity vs. translatability.

Compares how recoverable source language is from residual-stream activations
relative to the untranslatable/translatable label — at both last-token and
concept-token positions across layers.

Includes:
  - Normalized improvement over chance: (Acc - Chance) / (1 - Chance)
  - Balanced accuracy and AUROC with GroupKFold
  - Surface confound baselines (predicting label from word length, token count,
    language, category, and script without activations)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _encode_labels(values: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    uniq = sorted(set(values))
    mapping = {v: i for i, v in enumerate(uniq)}
    y = np.array([mapping[v] for v in values], dtype=np.int64)
    return y, mapping


def normalized_improvement(acc: float, chance: float) -> float:
    """Normalized improvement over chance: (acc - chance) / (1 - chance)."""
    if np.isnan(acc) or chance >= 1.0:
        return float("nan")
    return float((acc - chance) / (1.0 - chance))


def probe_layer(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 2000,
) -> Dict:
    """
    Logistic regression probe with concept-grouped CV when groups are provided.

    Using GroupKFold (by concept word) prevents train/test leakage from the
    three prompt templates that share the same concept label.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, label_binarize

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_classes = len(np.unique(y))
    chance = 1.0 / n_classes

    counts = np.bincount(y)
    min_count = int(counts.min()) if len(counts) else 0

    if groups is not None:
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))
        splits = min(n_splits, n_groups, min_count)
        if splits < 2:
            return {
                "accuracy_mean": float("nan"),
                "accuracy_std": float("nan"),
                "balanced_acc_mean": float("nan"),
                "auc_mean": float("nan"),
                "normalized_above_chance": float("nan"),
                "chance": chance,
                "n_classes": n_classes,
                "n_samples": int(len(y)),
                "n_splits": 0,
                "cv": "group",
                "error": "insufficient_groups",
            }
        fold_iter = GroupKFold(n_splits=splits).split(X, y, groups)
        cv_name = "group"
    else:
        splits = min(n_splits, min_count)
        if splits < 2:
            return {
                "accuracy_mean": float("nan"),
                "accuracy_std": float("nan"),
                "balanced_acc_mean": float("nan"),
                "auc_mean": float("nan"),
                "normalized_above_chance": float("nan"),
                "chance": chance,
                "n_classes": n_classes,
                "n_samples": int(len(y)),
                "n_splits": 0,
                "cv": "stratified",
                "error": "insufficient_samples_per_class",
            }
        fold_iter = StratifiedKFold(
            n_splits=splits, shuffle=True, random_state=seed
        ).split(X, y)
        cv_name = "stratified"

    accs = []
    b_accs = []
    aucs = []

    for train_idx, test_idx in fold_iter:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            random_state=seed,
            C=1.0,
        )
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test)
        accs.append(accuracy_score(y[test_idx], pred))
        b_accs.append(balanced_accuracy_score(y[test_idx], pred))

        try:
            if n_classes == 2:
                proba = clf.predict_proba(X_test)[:, 1]
                aucs.append(roc_auc_score(y[test_idx], proba))
            else:
                proba = clf.predict_proba(X_test)
                y_bin = label_binarize(y[test_idx], classes=np.arange(n_classes))
                if y_bin.shape[1] == n_classes and y_bin.sum() > 0:
                    aucs.append(
                        roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
                    )
        except Exception:
            pass

    mean_acc = float(np.mean(accs))
    return {
        "accuracy_mean": mean_acc,
        "accuracy_std": float(np.std(accs)),
        "balanced_acc_mean": float(np.mean(b_accs)),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "normalized_above_chance": normalized_improvement(mean_acc, chance),
        "chance": float(chance),
        "n_classes": int(n_classes),
        "n_samples": int(len(y)),
        "n_splits": int(splits),
        "cv": cv_name,
    }


def probe_translatability_within_language(
    X: np.ndarray,
    languages: Sequence[str],
    translatability: Sequence[str],
    groups: Optional[Sequence[str]] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Same-language-only binary probe for translatability (removes language confound).
    """
    languages = list(languages)
    translatability = list(translatability)
    X = np.asarray(X)
    per_lang = {}
    accs = []
    b_accs = []

    for lang in sorted(set(languages)):
        idx = [i for i, l in enumerate(languages) if l == lang]
        y_raw = [translatability[i] for i in idx]
        if len(set(y_raw)) < 2:
            continue
        y, _ = _encode_labels(y_raw)
        g = None
        if groups is not None:
            g = np.array([groups[i] for i in idx])
        res = probe_layer(X[idx], y, groups=g, n_splits=n_splits, seed=seed)
        if res.get("n_splits", 0) >= 2 and not np.isnan(res["accuracy_mean"]):
            per_lang[lang] = res
            accs.append(res["accuracy_mean"])
            b_accs.append(res.get("balanced_acc_mean", res["accuracy_mean"]))

    mean_acc = float(np.mean(accs)) if accs else float("nan")
    return {
        "by_language": per_lang,
        "accuracy_mean": mean_acc,
        "accuracy_std": float(np.std(accs)) if accs else float("nan"),
        "balanced_acc_mean": float(np.mean(b_accs)) if b_accs else float("nan"),
        "normalized_above_chance": normalized_improvement(mean_acc, 0.5),
        "n_languages": len(per_lang),
        "chance": 0.5,
    }


def confound_baseline_probe(
    metadata: List[Dict],
    keep_mask: Optional[Sequence[bool]] = None,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Fit logistic regression predicting translatability using ONLY surface features:
      - word length (characters)
      - token count (whitespace words)
      - language (one-hot)
      - category (one-hot)
    Uses the exact same GroupKFold setup.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    if keep_mask is not None:
        mask = np.asarray(keep_mask, dtype=bool)
        meta = [m for m, k in zip(metadata, mask) if k]
    else:
        meta = metadata

    X_num = []
    X_cat = []
    y = []
    groups = []

    for m in meta:
        w = m["concept_word"]
        char_len = len(w)
        tok_count = len(w.split())
        X_num.append([char_len, tok_count])
        X_cat.append([m["language"], m["category"]])
        y.append(1 if m["translatability"] == "untranslatable" else 0)
        groups.append(f"{m['language']}::{m['concept_word']}")

    X_num = np.array(X_num, dtype=float)
    enc = OneHotEncoder(sparse_output=False, drop="first")
    X_cat_enc = enc.fit_transform(X_cat)
    X = np.hstack([X_num, X_cat_enc])
    y = np.array(y, dtype=int)
    groups = np.array(groups)

    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    accs, b_accs, aucs = [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_tr, y[train_idx])
        pred = clf.predict(X_te)
        prob = clf.predict_proba(X_te)[:, 1]

        accs.append(accuracy_score(y[test_idx], pred))
        b_accs.append(balanced_accuracy_score(y[test_idx], pred))
        try:
            aucs.append(roc_auc_score(y[test_idx], prob))
        except Exception:
            pass

    mean_acc = float(np.mean(accs))
    return {
        "accuracy_mean": mean_acc,
        "accuracy_std": float(np.std(accs)),
        "balanced_acc_mean": float(np.mean(b_accs)),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "normalized_above_chance": normalized_improvement(mean_acc, 0.5),
        "chance": 0.5,
        "n_samples": len(y),
        "features": ["char_len", "token_count", "language", "category"],
    }


def run_probe_suite(
    hidden_states: Dict[int, np.ndarray],
    metadata: List[Dict],
    keep_mask: Optional[Sequence[bool]] = None,
    layer_stride: int = 1,
    n_splits: int = 5,
    seed: int = 42,
    within_language: bool = True,
) -> Dict:
    """
    Run language and translatability probes at every layer (optionally strided).

    Uses GroupKFold by concept_word to avoid template leakage.
    """
    if keep_mask is not None:
        mask = np.asarray(keep_mask, dtype=bool)
    else:
        mask = np.ones(len(metadata), dtype=bool)

    meta = [m for m, k in zip(metadata, mask) if k]
    languages = [m["language"] for m in meta]
    transl = [m["translatability"] for m in meta]
    concept_ids = [f"{m['language']}::{m['concept_word']}" for m in meta]
    y_lang, lang_map = _encode_labels(languages)
    y_trans, trans_map = _encode_labels(transl)
    groups = np.array(concept_ids)

    layer_indices = sorted(hidden_states.keys())
    layer_indices = layer_indices[:: max(1, layer_stride)]

    language_by_layer = {}
    transl_by_layer = {}
    within_by_layer = {}

    for li in layer_indices:
        X_full = hidden_states[li]
        X = X_full[mask]
        language_by_layer[str(li)] = probe_layer(
            X, y_lang, groups=groups, n_splits=n_splits, seed=seed
        )
        transl_by_layer[str(li)] = probe_layer(
            X, y_trans, groups=groups, n_splits=n_splits, seed=seed
        )
        if within_language:
            within_by_layer[str(li)] = probe_translatability_within_language(
                X, languages, transl, groups=concept_ids, n_splits=n_splits, seed=seed
            )

    def _peak(curve: Dict[str, Dict], key: str = "accuracy_mean") -> Dict:
        best_l, best_v = None, -1.0
        best_norm = -1.0
        for l, res in curve.items():
            v = res.get(key, float("nan"))
            if v is not None and not np.isnan(v) and v > best_v:
                best_v, best_l = v, int(l)
                best_norm = res.get("normalized_above_chance", float("nan"))
        return {"layer": best_l, "accuracy": best_v, "normalized_above_chance": best_norm}

    confound_res = confound_baseline_probe(metadata, keep_mask=keep_mask, n_splits=n_splits, seed=seed)

    return {
        "n_prompts_kept": int(mask.sum()),
        "n_prompts_total": int(len(metadata)),
        "language_label_map": lang_map,
        "translatability_label_map": trans_map,
        "language_probe": language_by_layer,
        "translatability_probe": transl_by_layer,
        "translatability_within_language": within_by_layer,
        "confound_baseline": confound_res,
        "peak_language": _peak(language_by_layer),
        "peak_translatability": _peak(transl_by_layer),
        "peak_within_language": _peak(within_by_layer) if within_by_layer else {},
        "chance_language": language_by_layer[str(layer_indices[0])]["chance"]
        if layer_indices
        else None,
        "chance_translatability": 0.5,
        "cv": "GroupKFold by concept_word",
    }


def summarize_probe_contrast(suite: Dict) -> Dict:
    """Compact summary: normalized improvements and confound comparison."""
    pl = suite.get("peak_language", {})
    pt = suite.get("peak_translatability", {})
    pw = suite.get("peak_within_language", {})
    cb = suite.get("confound_baseline", {})

    lang_norm = pl.get("normalized_above_chance")
    trans_norm = pt.get("normalized_above_chance")
    within_norm = pw.get("normalized_above_chance")
    cb_acc = cb.get("accuracy_mean")
    cb_norm = cb.get("normalized_above_chance")

    return {
        "peak_language_acc": pl.get("accuracy"),
        "peak_language_layer": pl.get("layer"),
        "peak_language_normalized": lang_norm,
        "peak_translatability_acc": pt.get("accuracy"),
        "peak_translatability_layer": pt.get("layer"),
        "peak_translatability_normalized": trans_norm,
        "peak_within_language_acc": pw.get("accuracy"),
        "peak_within_language_normalized": within_norm,
        "confound_baseline_acc": cb_acc,
        "confound_baseline_auroc": cb.get("auc_mean"),
        "confound_baseline_normalized": cb_norm,
        "normalized_gap_lang_minus_trans": (
            None
            if lang_norm is None or trans_norm is None
            else float(lang_norm - trans_norm)
        ),
        "activation_advantage_over_confound": (
            None
            if pt.get("accuracy") is None or cb_acc is None
            else float(pt["accuracy"] - cb_acc)
        ),
    }
