#!/usr/bin/env python3
"""
Step 3: Run all analyses on extracted activations.

Performs:
    1. Logit Lens analysis (English Routing Rate)
    2. PCA + UMAP dimensionality reduction
    3. K-means clustering evaluation
    4. Activation trajectory analysis
    5. Statistical tests

Usage:
    python scripts/03_run_analysis.py
    python scripts/03_run_analysis.py --activations-dir data/activations --output-dir data/analysis
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import (
    load_activations,
    load_logits,
    load_metadata,
    load_lm_head,
    save_results,
)
from src.analysis.logit_lens import aggregate_err_by_translatability
from src.analysis.clustering import (
    apply_pca,
    apply_umap,
    cluster_kmeans,
    evaluate_clustering,
    compute_silhouette_by_group,
    cluster_per_layer,
)
from src.analysis.trajectory import (
    compute_translatable_centroids,
    compute_trajectory_distances,
    compute_per_language_trajectories,
    find_peak_distance_layer,
)
from src.analysis.statistics import (
    run_all_statistical_tests,
    format_results_table,
)
from src.analysis.cross_model import (
    load_model_results,
    build_cross_model_report,
)


def run_analysis(
    activations_dir: str = "data/activations",
    output_dir: str = "data/analysis",
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    pca_components: int = 50,
    n_clusters: int = 10,
    model_key: str = "llama-3-8b",
):
    """Run the complete analysis pipeline."""
    activations_dir = os.path.join(activations_dir, model_key)
    output_dir = os.path.join(output_dir, model_key)
    os.makedirs(output_dir, exist_ok=True)

    print("Loading activations...")
    hidden_states = load_activations(activations_dir)
    logits = load_logits(activations_dir)
    metadata = load_metadata(activations_dir)

    print(f"  {len(hidden_states)} layers loaded")
    print(f"  {len(metadata)} prompts")

    all_results = {}

    # ─── 1. Logit Lens ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. Logit Lens Analysis")
    print("=" * 60)

    # Load LM head for unembedding
    lm_head_weights, lm_head_bias = load_lm_head(model_name)

    if lm_head_weights is not None and logits is not None:
        err_by_translatability = aggregate_err_by_translatability(
            hidden_states=hidden_states,
            logits=logits,
            metadata=metadata,
            tokenizer=None,  # Will compute ERR using weights directly
            lm_head_weights=lm_head_weights,
            lm_head_bias=lm_head_bias,
        )

        # Note: We compute ERR using the unembedding projection
        # For each layer, project hidden states through LM head and check
        # if top-1 token is English
        all_results["logit_lens"] = {
            "err_by_translatability": {
                k: {str(kk): vv for kk, vv in v.items()}
                for k, v in err_by_translatability.items()
            }
        }
        print(f"  ERR computed for {len(err_by_translatability)} groups")
    else:
        print("  Skipping logit lens (LM head not available)")

    # ─── 2. Clustering Analysis ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. Clustering Analysis")
    print("=" * 60)

    # Use a representative layer for initial analysis (middle layer)
    all_layer_indices = sorted(hidden_states.keys())
    mid_layer = all_layer_indices[len(all_layer_indices) // 2]
    print(f"  Using layer {mid_layer} for initial clustering")

    hs_mid = hidden_states[mid_layer]

    # PCA
    print("\n  PCA reduction...")
    hs_pca, pca_model = apply_pca(hs_mid, n_components=min(pca_components, hs_mid.shape[1]))

    # UMAP
    print("\n  UMAP embedding...")
    umap_2d = apply_umap(hs_pca, n_components=2)

    # K-means clustering
    print("\n  K-means clustering...")
    true_language_labels = np.array([m["language_code"] for m in metadata])
    pred_labels = cluster_kmeans(hs_pca, n_clusters=n_clusters)

    # Evaluate
    metrics = evaluate_clustering(pred_labels, true_language_labels, hs_pca)
    print(f"\n  Clustering metrics at layer {mid_layer}:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # Silhouette by group
    silhouette_by_group = compute_silhouette_by_group(
        hs_pca,
        pred_labels,
        np.array([m["translatability"] for m in metadata]),
    )
    print(f"\n  Silhouette by translatability:")
    for group, score in silhouette_by_group.items():
        print(f"    {group}: {score:.4f}")

    # Cluster per layer
    print("\n  Clustering across all layers...")
    ari_by_layer = {}
    for layer_idx, hs in hidden_states.items():
        hs_pca_l, _ = apply_pca(hs, n_components=min(pca_components, hs.shape[1]))
        pred_l = cluster_kmeans(hs_pca_l, n_clusters=n_clusters)
        metrics_l = evaluate_clustering(pred_l, true_language_labels, hs_pca_l)
        ari_by_layer[layer_idx] = metrics_l["adjusted_rand_index"]

    all_results["clustering"] = {
        "umap_2d": umap_2d.tolist(),
        "silhouette_by_group": silhouette_by_group,
        "ari_by_layer": {str(k): v for k, v in ari_by_layer.items()},
        "mid_layer_metrics": {k: float(v) for k, v in metrics.items()},
    }

    # ─── 3. Trajectory Analysis ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. Trajectory Analysis")
    print("=" * 60)

    # Compute translatable centroids as reference
    translatable_centroids = compute_translatable_centroids(hidden_states, metadata)

    # Distance from translatable centroid by translatability
    distances_by_translatability = compute_trajectory_distances(
        hidden_states, metadata, translatable_centroids
    )

    print("\n  Distance from translatable centroid:")
    for group, layer_dists in distances_by_translatability.items():
        values = list(layer_dists.values())
        print(f"    {group}: mean={np.mean(values):.4f}, max={np.max(values):.4f}")

    # Per-language trajectories
    distances_by_language = compute_per_language_trajectories(
        hidden_states, metadata, translatable_centroids
    )

    # Find peak distance layers
    peak_distances = {}
    for group, layer_dists in distances_by_translatability.items():
        peak_layer, peak_dist = find_peak_distance_layer(layer_dists)
        peak_distances[group] = {"layer": peak_layer, "distance": peak_dist}
        print(f"    {group} peak: layer {peak_layer} (dist={peak_dist:.4f})")

    all_results["trajectory"] = {
        "distances_by_translatability": {
            k: {str(kk): vv for kk, vv in v.items()}
            for k, v in distances_by_translatability.items()
        },
        "distances_by_language": {
            k: {str(kk): vv for kk, vv in v.items()}
            for k, v in distances_by_language.items()
        },
        "peak_distances": peak_distances,
    }

    # ─── 4. Statistical Tests ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4. Statistical Tests")
    print("=" * 60)

    # Prepare data for tests
    err_by_trans = all_results.get("logit_lens", {}).get("err_by_translatability", {})
    sil_by_trans = all_results.get("clustering", {}).get("silhouette_by_group", {})
    peak_dist = all_results.get("trajectory", {}).get("peak_distances", {})

    # Convert peak distances to arrays
    peak_arrays = {}
    for group in ["untranslatable", "translatable"]:
        if group in peak_dist:
            peak_arrays[group] = [peak_dist[group]["distance"]]
        else:
            peak_arrays[group] = []

    stat_results = run_all_statistical_tests(
        err_by_translatability=err_by_trans,
        silhouette_by_translatability=sil_by_trans,
        trajectory_by_translatability=distances_by_translatability,
        peak_distance_by_translatability=peak_arrays,
    )

    all_results["statistics"] = stat_results

    print("\n" + format_results_table(stat_results))

    # ─── Save Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    save_results(all_results, os.path.join(output_dir, "analysis_results.json"))

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


def main():
    import argparse
    from src.extraction.model_config import get_model_config, resolve_model_keys

    parser = argparse.ArgumentParser(description="Run activation analysis")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys or 'all' (default: llama-3-8b)")
    parser.add_argument("--activations-dir", default="data/activations")
    parser.add_argument("--output-dir", default="data/analysis")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model name (legacy)")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--n-clusters", type=int, default=10)
    args = parser.parse_args()

    model_keys = resolve_model_keys(args.models)

    all_model_results = {}
    for mk in model_keys:
        cfg = get_model_config(mk)
        print("\n" + "=" * 60)
        print(f"Running analysis for: {mk} ({cfg['short_name']})")
        print("=" * 60)

        run_analysis(
            activations_dir=args.activations_dir,
            output_dir=args.output_dir,
            model_name=cfg["hf_name"],
            pca_components=args.pca_components,
            n_clusters=args.n_clusters,
            model_key=mk,
        )

    # Cross-model comparison
    if len(model_keys) > 1:
        print("\n" + "=" * 60)
        print("Running Cross-Model Comparison")
        print("=" * 60)

        model_results = load_model_results(args.output_dir, model_keys)
        report = build_cross_model_report(model_results)

        from src.utils.io import save_results
        save_results(report, os.path.join(args.output_dir, "cross_model.json"))
        print(f"  Cross-model comparison: {len(report['models_analyzed'])} models")
        if "err_correlations" in report:
            for pair, corr in report["err_correlations"].items():
                print(f"    {pair}: r={corr['pearson_r']:.3f} ({corr['n_layers']} layers)")


if __name__ == "__main__":
    main()
