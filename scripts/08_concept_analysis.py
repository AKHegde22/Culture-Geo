#!/usr/bin/env python3
"""
Step 8: Concept-token position analysis (Step 3 of the measurement fixes).

Runs the English-probability, clustering, and trajectory analyses on the
hidden states extracted at the concept word's own token position, where the
concept itself (not the whole sentence) is represented.

Usage:
    python scripts/08_concept_analysis.py --models all
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.model_config import get_model_config, resolve_model_keys
from src.utils.io import save_results


def run_concept_analysis(
    activations_dir: str,
    output_dir: str,
    model_name: str,
    n_clusters: int = 10,
    pca_components: int = 50,
    device: str = None,
) -> dict:
    from src.utils.io import (
        load_concept_activations,
        load_lm_head_from_file,
        load_final_norm_from_file,
        load_metadata,
    )
    from transformers import AutoTokenizer

    from src.analysis.english_probability import (
        compute_english_probability_mass,
        mid_layer_indices,
        welch_ttest,
        permutation_test_by_layer,
    )
    from src.analysis.clustering import (
        apply_pca,
        cluster_kmeans,
        evaluate_clustering,
        compute_silhouette_by_group,
    )
    from src.analysis.trajectory import (
        compute_translatable_centroids,
        compute_trajectory_distances,
        find_peak_distance_layer,
    )

    print("  Loading concept-token activations...")
    hidden_states = load_concept_activations(activations_dir)
    if not hidden_states:
        raise RuntimeError("No concept-layer activations found; run extraction first")
    metadata = load_metadata(activations_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lm_head_weights, lm_head_bias = load_lm_head_from_file(
        os.path.join(activations_dir, "lm_head.npz")
    )
    final_norm_gamma = load_final_norm_from_file(
        os.path.join(activations_dir, "final_norm.npz")
    )

    group_mask = np.array(
        [m["translatability"] == "untranslatable" for m in metadata], dtype=bool
    )
    layer_indices = sorted(hidden_states.keys())

    results = {"n_prompts": len(metadata), "layer_indices": layer_indices}

    # ─── English probability at concept position ────────────────────────────
    print("  English probability mass at concept position...")
    p_eng, p_layers = compute_english_probability_mass(
        hidden_states, tokenizer, lm_head_weights, lm_head_bias, final_norm_gamma,
        device=device,
    )
    mid_layers = mid_layer_indices(p_layers)
    mid_rows = [p_layers.index(l) for l in mid_layers]
    per_prompt_mid = p_eng[mid_rows].mean(axis=0)

    results["english_probability"] = {
        "mid_layer_indices": mid_layers,
        "per_layer_diff": {
            str(p_layers[i]): float(p_eng[i][group_mask].mean() - p_eng[i][~group_mask].mean())
            for i in range(len(p_layers))
        },
        "permutation_p_by_layer": permutation_test_by_layer(
            p_eng, group_mask, p_layers
        ),
        "welch_midlayer": welch_ttest(per_prompt_mid[group_mask], per_prompt_mid[~group_mask]),
    }

    # ─── Clustering per layer ───────────────────────────────────────────────
    print("  Clustering per layer (ARI vs language)...")
    true_language_labels = np.array([m["language_code"] for m in metadata])
    ari_by_layer = {}
    for layer_idx in layer_indices:
        hs_pca, _ = apply_pca(
            hidden_states[layer_idx], n_components=min(pca_components, hidden_states[layer_idx].shape[1])
        )
        pred = cluster_kmeans(hs_pca, n_clusters=n_clusters)
        metrics = evaluate_clustering(pred, true_language_labels, hs_pca)
        ari_by_layer[layer_idx] = metrics["adjusted_rand_index"]

    mid_layer = layer_indices[len(layer_indices) // 2]
    hs_pca, _ = apply_pca(
        hidden_states[mid_layer], n_components=min(pca_components, hidden_states[mid_layer].shape[1])
    )
    pred = cluster_kmeans(hs_pca, n_clusters=n_clusters)
    sil_by_group = compute_silhouette_by_group(
        hs_pca, pred, np.array([m["translatability"] for m in metadata])
    )

    results["clustering"] = {
        "ari_by_layer": {str(k): float(v) for k, v in ari_by_layer.items()},
        "silhouette_by_group": {k: float(v) for k, v in sil_by_group.items()},
        "mid_layer": mid_layer,
    }

    # ─── Trajectory from translatable centroid ──────────────────────────────
    print("  Trajectory distances from translatable centroid...")
    centroids = compute_translatable_centroids(hidden_states, metadata)
    dists = compute_trajectory_distances(hidden_states, metadata, centroids)
    peaks = {}
    for group, layer_dists in dists.items():
        layer, dist = find_peak_distance_layer(layer_dists)
        peaks[group] = {"layer": layer, "distance": dist}

    results["trajectory"] = {
        "distances_by_translatability": {
            k: {str(kk): float(vv) for kk, vv in v.items()} for k, v in dists.items()
        },
        "peak_distances": peaks,
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--activations-dir", default="data/activations")
    parser.add_argument("--output-dir", default="data/analysis")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=10)
    args = parser.parse_args()

    for mk in resolve_model_keys(args.models):
        cfg = get_model_config(mk)
        print("=" * 70)
        print(f"Concept-position analysis: {mk} ({cfg['short_name']})")
        print("=" * 70)

        act_dir = os.path.join(args.activations_dir, mk)
        out_dir = os.path.join(args.output_dir, mk)
        os.makedirs(out_dir, exist_ok=True)

        results = run_concept_analysis(act_dir, out_dir, cfg["hf_name"],
                                       args.n_clusters, args.pca_components)
        save_results(results, os.path.join(out_dir, "concept_analysis.json"))

        w = results["english_probability"]["welch_midlayer"]
        print(f"\n  Concept-position mid-layer Welch: d={w['cohens_d']:.3f} "
              f"p={w['p_value']:.4f} (n={w['n_a']}/{w['n_b']})")
        sil = results["clustering"]["silhouette_by_group"]
        print(f"  Silhouette mid-layer: {sil}")
        print(f"  Trajectory peaks: {results['trajectory']['peak_distances']}")


if __name__ == "__main__":
    main()
