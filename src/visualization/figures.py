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


def generate_all_figures(
    results_dir: str = "data/analysis",
    output_dir: str = "paper/figures",
    activations_dir: str = "data/activations",
):
    """Generate all paper figures from saved results."""
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results[fname.replace(".json", "")] = json.load(f)

    print(f"Loaded {len(results)} result files")

    # Figure 1: UMAP embedding
    if "clustering" in results:
        clustering = results["clustering"]
        if "umap_2d" in clustering:
            umap_data = np.array(clustering["umap_2d"])

            # Need metadata - load from activations
            metadata_path = os.path.join(activations_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)

                # Plot by translatability
                plot_umap_embedding(
                    umap_data,
                    metadata,
                    color_by="translatability",
                    output_path=os.path.join(output_dir, "fig1_umap_translatability.pdf"),
                    title="(a) UMAP: Translatable vs. Untranslatable Concepts",
                )

                # Plot by language
                plot_umap_embedding(
                    umap_data,
                    metadata,
                    color_by="language",
                    output_path=os.path.join(output_dir, "fig1b_umap_language.pdf"),
                    title="(b) UMAP: Concepts by Source Language",
                )

    # Figure 2: English Routing Rate
    if "logit_lens" in results:
        lens = results["logit_lens"]
        if "err_by_translatability" in lens:
            err_data = {
                k: {int(kk): vv for kk, vv in v.items()}
                for k, v in lens["err_by_translatability"].items()
            }
            plot_err_across_layers(
                err_data,
                output_path=os.path.join(output_dir, "fig2_err_across_layers.pdf"),
                title="English Routing Rate Across Model Layers",
            )

    # Figure 3: Trajectory distances
    if "trajectory" in results:
        traj = results["trajectory"]
        if "distances_by_translatability" in traj:
            dist_data = {
                k: {int(kk): vv for kk, vv in v.items()}
                for k, v in traj["distances_by_translatability"].items()
            }
            plot_trajectory_distances(
                dist_data,
                output_path=os.path.join(output_dir, "fig3_trajectory_distances.pdf"),
                title="Distance from English-Centric Centroid Across Layers",
            )

    # Figure 4: ARI across layers
    if "clustering" in results:
        clustering = results["clustering"]
        if "ari_by_layer" in clustering:
            ari_data = {int(k): v for k, v in clustering["ari_by_layer"].items()}
            plot_layerwise_ari(
                ari_data,
                output_path=os.path.join(output_dir, "fig4_ari_by_layer.pdf"),
                title="Clustering Quality (ARI) Across Model Layers",
            )

    # Figure 5: Silhouette scores
    if "clustering" in results:
        clustering = results["clustering"]
        if "silhouette_by_group" in clustering:
            plot_silhouette_by_group(
                clustering["silhouette_by_group"],
                output_path=os.path.join(output_dir, "fig5_silhouette_by_group.pdf"),
                title="Cluster Separation by Translatability",
            )

    # Figure 6: Statistical results
    if "statistics" in results:
        plot_statistical_results(
            results["statistics"],
            output_path=os.path.join(output_dir, "fig6_statistical_results.pdf"),
            title="Effect Sizes: Untranslatable vs. Translatable",
        )

    # List generated figures
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
