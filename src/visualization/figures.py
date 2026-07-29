"""
Generate all paper figures from analysis results.

Usage:
    python scripts/04_generate_figures.py --results-dir data/analysis --output-dir paper/figures
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.plots import (
    plot_umap_embedding,
    plot_err_across_layers,
    plot_trajectory_distances,
    plot_layerwise_ari,
    plot_silhouette_by_group,
    plot_concept_heatmap,
    plot_statistical_results,
)


MODEL_KEYS = ["llama-3-8b", "llama-3-8b-instruct", "mistral-7b", "qwen-2.5-7b"]

MODEL_SHORT_NAMES = {
    "llama-3-8b": "llama3-8b",
    "llama-3-8b-instruct": "llama3-8b-it",
    "mistral-7b": "mistral-7b",
    "qwen-2.5-7b": "qwen2.5-7b",
}


def generate_figures_for_model(
    model_key: str,
    results_dir: str,
    activations_dir: str,
    output_dir: str,
):
    prefix = MODEL_SHORT_NAMES.get(model_key, model_key)
    model_results_dir = os.path.join(results_dir, model_key)
    model_activations_dir = os.path.join(activations_dir, model_key)

    if not os.path.exists(model_results_dir):
        print(f"  Skipping {model_key}: no results found")
        return

    results = {}
    for fname in os.listdir(model_results_dir):
        if fname.endswith(".json"):
            with open(os.path.join(model_results_dir, fname)) as f:
                results[fname.replace(".json", "")] = json.load(f)

    print(f"  Generating figures for {model_key}...")

    if "clustering" in results:
        clustering = results["clustering"]
        if "umap_2d" in clustering:
            umap_data = np.array(clustering["umap_2d"])
            metadata_path = os.path.join(model_activations_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                plot_umap_embedding(
                    umap_data, metadata,
                    color_by="translatability",
                    output_path=os.path.join(output_dir, f"{prefix}_umap_translatability.pdf"),
                    title=f"({model_key}) UMAP by Translatability",
                )
                plot_umap_embedding(
                    umap_data, metadata,
                    color_by="language",
                    output_path=os.path.join(output_dir, f"{prefix}_umap_language.pdf"),
                    title=f"({model_key}) UMAP by Language",
                )

        if "ari_by_layer" in clustering:
            ari_data = {int(k): v for k, v in clustering["ari_by_layer"].items()}
            plot_layerwise_ari(
                ari_data,
                output_path=os.path.join(output_dir, f"{prefix}_ari_by_layer.pdf"),
                title=f"({model_key}) Clustering Quality (ARI) Across Layers",
            )

        if "silhouette_by_group" in clustering:
            plot_silhouette_by_group(
                clustering["silhouette_by_group"],
                output_path=os.path.join(output_dir, f"{prefix}_silhouette_by_group.pdf"),
                title=f"({model_key}) Cluster Separation by Translatability",
            )

    if "logit_lens" in results:
        lens = results["logit_lens"]
        if "err_by_translatability" in lens:
            err_data = {
                k: {int(kk): vv for kk, vv in v.items()}
                for k, v in lens["err_by_translatability"].items()
            }
            plot_err_across_layers(
                err_data,
                output_path=os.path.join(output_dir, f"{prefix}_err_across_layers.pdf"),
                title=f"({model_key}) English Routing Rate Across Layers",
            )

    if "trajectory" in results:
        traj = results["trajectory"]
        if "distances_by_translatability" in traj:
            dist_data = {
                k: {int(kk): vv for kk, vv in v.items()}
                for k, v in traj["distances_by_translatability"].items()
            }
            plot_trajectory_distances(
                dist_data,
                output_path=os.path.join(output_dir, f"{prefix}_trajectory_distances.pdf"),
                title=f"({model_key}) Distance from Centroid Across Layers",
            )

    if "statistics" in results:
        plot_statistical_results(
            results["statistics"],
            output_path=os.path.join(output_dir, f"{prefix}_statistical_results.pdf"),
            title=f"({model_key}) Effect Sizes: Untranslatable vs. Translatable",
        )


def generate_cross_model_figures(
    results_dir: str,
    output_dir: str,
    model_keys: List[str],
):
    cross_model_results = {}
    for mk in model_keys:
        path = os.path.join(results_dir, mk, "analysis_results.json")
        if os.path.exists(path):
            with open(path) as f:
                cross_model_results[mk] = json.load(f)

    if not cross_model_results:
        print("  No cross-model data available")
        return

    from src.analysis.cross_model import (
        collect_err_curves,
        collect_ari_curves,
        collect_silhouette_scores,
        build_cross_model_report,
    )

    err_curves = collect_err_curves(cross_model_results)
    if err_curves:
        plot_cross_model_err(
            err_curves,
            output_path=os.path.join(output_dir, "cross_model_err.pdf"),
            title="English Routing Rate: All Models",
        )

    ari_curves = collect_ari_curves(cross_model_results)
    if ari_curves:
        plot_cross_model_ari(
            ari_curves,
            output_path=os.path.join(output_dir, "cross_model_ari.pdf"),
            title="Adjusted Rand Index: All Models",
        )

    sil_scores = collect_silhouette_scores(cross_model_results)
    if sil_scores:
        plot_cross_model_silhouette(
            sil_scores,
            output_path=os.path.join(output_dir, "cross_model_silhouette.pdf"),
            title="Silhouette Scores: All Models",
        )

    report = build_cross_model_report(cross_model_results)
    report_path = os.path.join(output_dir, "cross_model_comparison.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Cross-model comparison saved to {report_path}")


def generate_all_figures(
    results_dir: str = "data/analysis",
    output_dir: str = "paper/figures",
    activations_dir: str = "data/activations",
    model_keys: Optional[List[str]] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    if model_keys is None:
        model_keys = MODEL_KEYS

    for mk in model_keys:
        generate_figures_for_model(mk, results_dir, activations_dir, output_dir)

    if len(model_keys) > 1:
        print("\nGenerating cross-model figures...")
        generate_cross_model_figures(results_dir, output_dir, model_keys)

    figures = sorted(os.listdir(output_dir))
    print(f"\nGenerated {len(figures)} figures:")
    for fig in figures:
        size = os.path.getsize(os.path.join(output_dir, fig))
        print(f"  - {fig} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="data/analysis")
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--activations-dir", default="data/activations")
    args = parser.parse_args()

    generate_all_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        activations_dir=args.activations_dir,
    )
