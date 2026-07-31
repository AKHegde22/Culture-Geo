"""
Visualization functions for the Untranslatability study.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns


# ─── Style Configuration ──────────────────────────────────────────────────

LANGUAGE_COLORS = {
    "ja": "#E74C3C",   # Red
    "de": "#3498DB",   # Blue
    "pt": "#2ECC71",   # Green
    "ko": "#9B59B6",   # Purple
    "ar": "#F39C12",   # Orange
    "hi": "#E67E22",   # Dark orange
    "ru": "#1ABC9C",   # Teal
    "da": "#34495E",   # Dark gray
    "es": "#E91E63",   # Pink
    "fr": "#00BCD4",   # Cyan
}

TRANSLATABILITY_COLORS = {
    "untranslatable": "#E74C3C",
    "translatable": "#3498DB",
}


def setup_style():
    """Set up matplotlib style."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("seaborn-whitegrid")

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_umap_embedding(
    umap_2d: np.ndarray,
    metadata: List[Dict],
    color_by: str = "translatability",
    output_path: Optional[str] = None,
    title: str = "UMAP Embedding of Concept Activations",
) -> plt.Figure:
    """
    Plot 2D UMAP embedding colored by language or translatability.

    Args:
        umap_2d: [n_prompts, 2] UMAP coordinates
        metadata: List of metadata dicts
        color_by: "translatability" or "language"
        output_path: Path to save figure
        title: Plot title
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    if color_by == "translatability":
        colors = [TRANSLATABILITY_COLORS[m["translatability"]] for m in metadata]
        ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=colors, alpha=0.6, s=30)

        # Legend
        patches = [
            mpatches.Patch(color=c, label=t)
            for t, c in TRANSLATABILITY_COLORS.items()
        ]
        ax.legend(handles=patches, title="Translatability")

    elif color_by == "language":
        languages = list(set(m["language_code"] for m in metadata))
        for lang in languages:
            indices = [i for i, m in enumerate(metadata) if m["language_code"] == lang]
            color = LANGUAGE_COLORS.get(lang, "#999999")
            label = metadata[indices[0]]["language"]
            ax.scatter(
                umap_2d[indices, 0],
                umap_2d[indices, 1],
                c=color,
                label=f"{label} ({lang})",
                alpha=0.6,
                s=30,
            )
        ax.legend(title="Language", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_err_across_layers(
    err_by_translatability: Dict[str, Dict[int, float]],
    output_path: Optional[str] = None,
    title: str = "English Routing Rate Across Layers",
) -> plt.Figure:
    """
    Plot English Routing Rate (ERR) at each layer for untranslatable vs translatable.

    Args:
        err_by_translatability: Dict mapping translatability -> {layer_idx: ERR}
        output_path: Path to save figure
        title: Plot title
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for translatability, layer_errs in err_by_translatability.items():
        layers = sorted(layer_errs.keys())
        values = [layer_errs[l] for l in layers]
        color = TRANSLATABILITY_COLORS.get(translatability, "#999999")
        ax.plot(layers, values, marker="o", color=color, label=translatability, linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("English Routing Rate")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_trajectory_distances(
    distances_by_group: Dict[str, Dict[int, float]],
    output_path: Optional[str] = None,
    title: str = "Cosine Distance from Reference Centroid Across Layers",
) -> plt.Figure:
    """
    Plot trajectory distances (from translatable centroid) across layers.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for group, layer_dists in distances_by_group.items():
        layers = sorted(layer_dists.keys())
        values = [layer_dists[l] for l in layers]
        color = TRANSLATABILITY_COLORS.get(group, "#999999")
        ax.plot(layers, values, marker="s", color=color, label=group, linewidth=2)
        ax.fill_between(layers, values, alpha=0.1, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Distance from Reference")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_layerwise_ari(
    ari_by_layer: Dict[int, float],
    output_path: Optional[str] = None,
    title: str = "Adjusted Rand Index Across Layers",
) -> plt.Figure:
    """
    Plot Adjusted Rand Index (clustering quality) at each layer.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = sorted(ari_by_layer.keys())
    values = [ari_by_layer[l] for l in layers]

    ax.bar(layers, values, color="#3498DB", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Mark the best layer
    best_layer = layers[np.argmax(values)]
    best_value = max(values)
    ax.annotate(
        f"Best: layer {best_layer}\nARI = {best_value:.3f}",
        xy=(best_layer, best_value),
        xytext=(best_layer + 3, best_value + 0.05),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_silhouette_by_group(
    silhouette_scores: Dict[str, float],
    output_path: Optional[str] = None,
    title: str = "Silhouette Scores by Translatability Group",
) -> plt.Figure:
    """
    Bar plot comparing silhouette scores between groups.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    groups = list(silhouette_scores.keys())
    values = [silhouette_scores[g] for g in groups]
    colors = [TRANSLATABILITY_COLORS.get(g, "#999999") for g in groups]

    bars = ax.bar(groups, values, color=colors, alpha=0.7)
    ax.set_ylabel("Silhouette Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_concept_heatmap(
    similarity_matrix: np.ndarray,
    concept_labels: List[str],
    layer_indices: List[int],
    output_path: Optional[str] = None,
    title: str = "Concept Similarity Across Layers",
) -> plt.Figure:
    """
    Heatmap of concept similarity across layers.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        similarity_matrix,
        xticklabels=concept_labels[:similarity_matrix.shape[1]],
        yticklabels=[f"Layer {l}" for l in layer_indices],
        cmap="RdYlBu_r",
        center=0,
        ax=ax,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_statistical_results(
    test_results: Dict,
    output_path: Optional[str] = None,
    title: str = "Statistical Test Results",
) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    test_names = []
    effects = []
    ci_lowers = []
    ci_uppers = []

    for name, result in test_results.items():
        if "error" in result:
            continue
        test_names.append(name.replace("_by_translatability", ""))
        effects.append(result.get("cohens_d", result.get("effect_size", 0)))
        ci_lowers.append(result.get("ci_95_lower", 0))
        ci_uppers.append(result.get("ci_95_upper", 0))

    if not test_names:
        ax.text(0.5, 0.5, "No significant results", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    y_pos = range(len(test_names))

    xerr = [
        [max(e - l, 0.0) for e, l in zip(effects, ci_lowers)],
        [max(u - e, 0.0) for e, u in zip(effects, ci_uppers)],
    ]
    ax.errorbar(
        effects,
        y_pos,
        xerr=xerr,
        fmt="o",
        color="#3498DB",
        capsize=5,
        markersize=8,
    )

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.2, color="green", linestyle=":", alpha=0.3, label="Small (d=0.2)")
    ax.axvline(x=0.5, color="orange", linestyle=":", alpha=0.3, label="Medium (d=0.5)")
    ax.axvline(x=0.8, color="red", linestyle=":", alpha=0.3, label="Large (d=0.8)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(test_names)
    ax.set_xlabel("Effect Size (Cohen's d)")
    ax.set_title(title)
    ax.legend(loc="lower right")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_cross_model_err(
    err_by_model: Dict[str, Dict[str, Dict[int, float]]],
    output_path: Optional[str] = None,
    title: str = "English Routing Rate Across Models",
) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_key, err_groups in err_by_model.items():
        color = TRANSLATABILITY_COLORS.get(model_key, "#999999")
        label = model_key
        for translatability, layer_errs in err_groups.items():
            layers = sorted(layer_errs.keys())
            values = [layer_errs[l] for l in layers]
            ls = "-" if translatability == "untranslatable" else "--"
            alpha = 1.0 if translatability == "untranslatable" else 0.5
            ax.plot(
                layers, values,
                marker="", linestyle=ls, color=color,
                label=f"{label} ({translatability})",
                linewidth=2, alpha=alpha,
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("English Routing Rate")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_cross_model_ari(
    ari_by_model: Dict[str, Dict[int, float]],
    output_path: Optional[str] = None,
    title: str = "Adjusted Rand Index Across Models",
) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_key, ari_layers in ari_by_model.items():
        color = TRANSLATABILITY_COLORS.get(model_key, "#999999")
        layers = sorted(ari_layers.keys())
        values = [ari_layers[l] for l in layers]
        ax.plot(layers, values, marker="o", color=color, label=model_key, linewidth=2, markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig


def plot_cross_model_silhouette(
    silhouette_by_model: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None,
    title: str = "Silhouette Scores by Model",
) -> plt.Figure:
    setup_style()
    model_keys = list(silhouette_by_model.keys())
    groups = ["untranslatable", "translatable"]

    fig, ax = plt.subplots(figsize=(8, 5))

    n_models = len(model_keys)
    bar_width = 0.35
    x = np.arange(n_models)

    for gi, group in enumerate(groups):
        values = []
        for mk in model_keys:
            v = silhouette_by_model.get(mk, {}).get(group, 0)
            values.append(v)
        offset = (gi - 0.5) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            label=group,
            color=TRANSLATABILITY_COLORS.get(group, "#999999"),
            alpha=0.8,
        )
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_keys, fontsize=9)
    ax.set_ylabel("Silhouette Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    return fig
