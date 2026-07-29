import json
import os
from typing import Dict, List, Optional

import numpy as np


MODEL_COLORS = {
    "llama-3-8b": "#E74C3C",
    "llama-3-8b-instruct": "#3498DB",
    "mistral-7b": "#2ECC71",
    "qwen-2.5-7b": "#9B59B6",
}

MODEL_LINESTYLES = {
    "llama-3-8b": "-",
    "llama-3-8b-instruct": "--",
    "mistral-7b": ":",
    "qwen-2.5-7b": "-.",
}


def load_model_results(
    base_analysis_dir: str,
    model_keys: List[str],
) -> Dict[str, Dict]:
    results = {}
    for key in model_keys:
        path = os.path.join(base_analysis_dir, key, "analysis_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[key] = json.load(f)
        else:
            print(f"  Warning: No analysis results found at {path}")
    return results


def collect_err_curves(
    model_results: Dict[str, Dict],
) -> Dict[str, Dict[str, Dict[int, float]]]:
    collected = {}
    for model_key, result in model_results.items():
        lens = result.get("logit_lens", {})
        err = lens.get("err_by_translatability", {})
        collected[model_key] = {
            group: {int(k): v for k, v in layer_errs.items()}
            for group, layer_errs in err.items()
        }
    return collected


def collect_ari_curves(
    model_results: Dict[str, Dict],
) -> Dict[str, Dict[int, float]]:
    collected = {}
    for model_key, result in model_results.items():
        clustering = result.get("clustering", {})
        ari = clustering.get("ari_by_layer", {})
        collected[model_key] = {int(k): v for k, v in ari.items()}
    return collected


def collect_silhouette_scores(
    model_results: Dict[str, Dict],
) -> Dict[str, Dict[str, float]]:
    collected = {}
    for model_key, result in model_results.items():
        clustering = result.get("clustering", {})
        sil = clustering.get("silhouette_by_group", {})
        if sil:
            collected[model_key] = sil
    return collected


def compute_err_correlation(
    err_curves: Dict[str, Dict[str, Dict[int, float]]],
    group: str = "untranslatable",
) -> Dict[str, Dict[str, float]]:
    correlations = {}
    model_keys = list(err_curves.keys())
    for i, mk1 in enumerate(model_keys):
        for mk2 in model_keys[i + 1:]:
            curve1 = err_curves.get(mk1, {}).get(group, {})
            curve2 = err_curves.get(mk2, {}).get(group, {})
            common_layers = sorted(set(curve1.keys()) & set(curve2.keys()))
            if len(common_layers) < 3:
                continue
            v1 = np.array([curve1[l] for l in common_layers])
            v2 = np.array([curve2[l] for l in common_layers])
            r = np.corrcoef(v1, v2)[0, 1]
            correlations[f"{mk1}_vs_{mk2}"] = {
                "model_a": mk1,
                "model_b": mk2,
                "pearson_r": float(r),
                "n_layers": len(common_layers),
            }
    return correlations


def build_cross_model_report(
    model_results: Dict[str, Dict],
) -> Dict:
    err_curves = collect_err_curves(model_results)
    ari_curves = collect_ari_curves(model_results)
    silhouette_scores = collect_silhouette_scores(model_results)
    err_corr = compute_err_correlation(err_curves)

    report = {
        "models_analyzed": list(model_results.keys()),
        "err_by_model": {
            mk: {
                group: {str(k): v for k, v in layers.items()}
                for group, layers in curves.items()
            }
            for mk, curves in err_curves.items()
        },
        "ari_by_model": {
            mk: {str(k): v for k, v in ari.items()}
            for mk, ari in ari_curves.items()
        },
        "silhouette_by_model": silhouette_scores,
        "err_correlations": err_corr,
    }
    return report
