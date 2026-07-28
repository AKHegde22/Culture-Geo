"""
Activation trajectory analysis.

Tracks how concept representations move through the model's layers,
measuring the distance from the English-centric centroid at each layer.

Key insight: If untranslatable concepts are "forced" through English,
they should show higher distance from English centroid in mid-layers
but potentially diverge in later layers.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_english_centroids(
    hidden_states_all: Dict[int, np.ndarray],
    metadata: List[Dict],
    language_filter: str = "en",
) -> Dict[int, np.ndarray]:
    """
    Compute the centroid of English-language activations at each layer.

    Args:
        hidden_states_all: Dict mapping layer_idx -> [n_prompts, d_model]
        metadata: List of metadata dicts for each prompt
        language_filter: Language code to compute centroid for

    Returns:
        Dict mapping layer_idx -> centroid [d_model]
    """
    # Find English prompts (if any exist)
    english_indices = [
        i for i, m in enumerate(metadata)
        if m.get("language_code") == language_filter
    ]

    if not english_indices:
        # If no English prompts, use all prompts as a baseline
        print(f"  Warning: No '{language_filter}' prompts found. Using all prompts as baseline.")
        english_indices = list(range(len(metadata)))

    centroids = {}
    for layer_idx, hs in hidden_states_all.items():
        centroids[layer_idx] = hs[english_indices].mean(axis=0)

    return centroids


def compute_translatable_centroids(
    hidden_states_all: Dict[int, np.ndarray],
    metadata: List[Dict],
) -> Dict[int, np.ndarray]:
    """
    Compute centroids for translatable concepts at each layer.

    These serve as the "English-adjacent" baseline since translatable
    concepts have direct English equivalents.
    """
    translatable_indices = [
        i for i, m in enumerate(metadata)
        if m["translatability"] == "translatable"
    ]

    centroids = {}
    for layer_idx, hs in hidden_states_all.items():
        centroids[layer_idx] = hs[translatable_indices].mean(axis=0)

    return centroids


def compute_trajectory_distances(
    hidden_states_all: Dict[int, np.ndarray],
    metadata: List[Dict],
    reference_centroids: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Compute cosine distances from reference centroids at each layer,
    broken down by translatability.

    Args:
        hidden_states_all: Dict mapping layer_idx -> [n_prompts, d_model]
        metadata: List of metadata dicts
        reference_centroids: Optional pre-computed centroids (if None, computes translatable centroids)

    Returns:
        Dict mapping translatability -> {layer_idx: mean_cosine_distance}
    """
    if reference_centroids is None:
        reference_centroids = compute_translatable_centroids(hidden_states_all, metadata)

    # Group indices by translatability
    groups = {"untranslatable": [], "translatable": []}
    for i, m in enumerate(metadata):
        groups[m["translatability"]].append(i)

    results = {}
    for translatability, indices in groups.items():
        layer_distances = {}
        for layer_idx, hs in hidden_states_all.items():
            if layer_idx not in reference_centroids:
                continue

            centroid = reference_centroids[layer_idx]
            # Normalize for cosine similarity
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

            distances = []
            for idx in indices:
                vec = hs[idx]
                vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
                cos_sim = np.dot(vec_norm, centroid_norm)
                distances.append(1.0 - cos_sim)  # Convert to distance

            layer_distances[layer_idx] = float(np.mean(distances))

        results[translatability] = layer_distances

    return results


def compute_per_language_trajectories(
    hidden_states_all: Dict[int, np.ndarray],
    metadata: List[Dict],
    reference_centroids: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Compute trajectory distances per language.

    Returns:
        Dict mapping language_code -> {layer_idx: mean_cosine_distance}
    """
    if reference_centroids is None:
        reference_centroids = compute_translatable_centroids(hidden_states_all, metadata)

    # Group by language
    lang_indices = {}
    for i, m in enumerate(metadata):
        lang = m["language_code"]
        if lang not in lang_indices:
            lang_indices[lang] = []
        lang_indices[lang].append(i)

    results = {}
    for lang, indices in lang_indices.items():
        layer_distances = {}
        for layer_idx, hs in hidden_states_all.items():
            if layer_idx not in reference_centroids:
                continue

            centroid = reference_centroids[layer_idx]
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

            distances = []
            for idx in indices:
                vec = hs[idx]
                vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
                cos_sim = np.dot(vec_norm, centroid_norm)
                distances.append(1.0 - cos_sim)

            layer_distances[layer_idx] = float(np.mean(distances))

        results[lang] = layer_distances

    return results


def compute_trajectory_curvature(
    distances: Dict[int, float],
) -> float:
    """
    Compute the "curvature" of a trajectory (how much it changes direction).

    High curvature = the trajectory changes direction frequently,
    suggesting the model is uncertain about how to represent the concept.
    """
    layers = sorted(distances.keys())
    if len(layers) < 3:
        return 0.0

    values = [distances[l] for l in layers]

    # Compute second derivatives (acceleration)
    accelerations = []
    for i in range(1, len(values) - 1):
        accel = values[i + 1] - 2 * values[i] + values[i - 1]
        accelerations.append(abs(accel))

    return float(np.mean(accelerations))


def find_peak_distance_layer(
    distances: Dict[int, float],
) -> Tuple[int, float]:
    """
    Find the layer where distance from reference centroid peaks.

    This indicates where the model diverges most from the reference
    (English-adjacent) representation.
    """
    if not distances:
        return -1, 0.0

    best_layer = max(distances, key=distances.get)
    return best_layer, distances[best_layer]


def compute_layerwise_similarity_matrix(
    hidden_states: Dict[int, np.ndarray],
    indices_a: List[int],
    indices_b: List[int],
) -> np.ndarray:
    """
    Compute cosine similarity between two sets of activations at each layer.

    Args:
        hidden_states: Dict mapping layer_idx -> [n_prompts, d_model]
        indices_a: Indices for first group
        indices_b: Indices for second group

    Returns:
        [n_layers, n_indices_a, n_indices_b] similarity matrix
    """
    layers = sorted(hidden_states.keys())
    n_a = len(indices_a)
    n_b = len(indices_b)

    similarity_matrix = np.zeros((len(layers), n_a, n_b))

    for li, layer_idx in enumerate(layers):
        hs = hidden_states[layer_idx]

        # Normalize
        hs_a = hs[indices_a]
        hs_b = hs[indices_b]
        hs_a_norm = hs_a / (np.linalg.norm(hs_a, axis=-1, keepdims=True) + 1e-8)
        hs_b_norm = hs_b / (np.linalg.norm(hs_b, axis=-1, keepdims=True) + 1e-8)

        # Compute pairwise cosine similarity
        similarity_matrix[li] = hs_a_norm @ hs_b_norm.T

    return similarity_matrix


def summarize_trajectories(
    trajectory_results: Dict[str, Dict[int, float]],
) -> str:
    """Generate text summary of trajectory analysis."""
    lines = ["Trajectory Analysis Summary"]
    lines.append("=" * 40)

    for translatability, layer_dists in trajectory_results.items():
        lines.append(f"\n  {translatability.upper()}:")
        layers = sorted(layer_dists.keys())
        values = [layer_dists[l] for l in layers]
        lines.append(f"    Mean distance: {np.mean(values):.4f}")
        lines.append(f"    Max distance:  {np.max(values):.4f} (layer {layers[np.argmax(values)]})")
        lines.append(f"    Min distance:  {np.min(values):.4f} (layer {layers[np.argmin(values)]})")

    return "\n".join(lines)
