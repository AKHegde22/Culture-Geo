"""
Clustering analysis of activation embeddings.

Applies PCA, UMAP, and k-means clustering to residual stream activations
to determine whether untranslatable concepts form distinct clusters.

Key metrics:
    - Adjusted Rand Index (ARI): agreement between cluster assignments and true language labels
    - Silhouette Score: how well-separated clusters are
    - Calinski-Harabasz Index: ratio of between-cluster to within-cluster dispersion
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
)


def apply_pca(
    activations: np.ndarray,
    n_components: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA]:
    """
    Apply PCA to reduce dimensionality of activations.

    Args:
        activations: [n_prompts, d_model]
        n_components: Number of principal components
        random_state: Random seed

    Returns:
        (transformed activations, fitted PCA object)
    """
    n_components = min(n_components, activations.shape[1], activations.shape[0])
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(activations)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {activations.shape[1]}d -> {n_components}d")
    print(f"  Explained variance: {explained_var:.3f}")

    return transformed, pca


def apply_umap(
    activations: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply UMAP for 2D visualization.

    Args:
        activations: [n_prompts, d_model] or [n_prompts, n_pca_components]
        n_neighbors: UMAP neighborhood size
        min_dist: UMAP minimum distance
        n_components: Output dimensions (2 for visualization)
        random_state: Random seed

    Returns:
        2D embedding [n_prompts, 2]
    """
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric="cosine",
    )
    embedding = reducer.fit_transform(activations)

    print(f"  UMAP: {activations.shape[1]}d -> {n_components}d")
    return embedding


def cluster_kmeans(
    activations: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """
    Apply k-means clustering.

    Args:
        activations: [n_prompts, n_features]
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Cluster assignments [n_prompts]
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(activations)
    return labels


def evaluate_clustering(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    activations: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate clustering quality against true labels.

    Args:
        predicted_labels: K-means cluster assignments
        true_labels: True language or translatability labels
        activations: The feature matrix used for clustering

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "adjusted_rand_index": adjusted_rand_score(true_labels, predicted_labels),
        "normalized_mutual_info": normalized_mutual_info_score(true_labels, predicted_labels),
        "silhouette_score": silhouette_score(activations, predicted_labels),
        "calinski_harabasz": calinski_harabasz_score(activations, predicted_labels),
    }
    return metrics


def find_optimal_k(
    activations: np.ndarray,
    k_range: range = range(2, 20),
    random_state: int = 42,
) -> Dict[int, Dict[str, float]]:
    """
    Find optimal number of clusters using multiple metrics.

    Args:
        activations: [n_prompts, n_features]
        k_range: Range of k values to test
        random_state: Random seed

    Returns:
        Dict mapping k -> metrics
    """
    results = {}
    for k in k_range:
        labels = cluster_kmeans(activations, n_clusters=k, random_state=random_state)
        sil = silhouette_score(activations, labels)
        ch = calinski_harabasz_score(activations, labels)
        results[k] = {
            "silhouette": sil,
            "calinski_harabasz": ch,
        }
        print(f"  k={k:2d}: silhouette={sil:.4f}, calinski_harabasz={ch:.1f}")

    return results


def compute_silhouette_by_group(
    activations: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute silhouette scores separately for each group.

    Args:
        activations: [n_prompts, n_features]
        labels: Cluster assignments
        group_labels: Group labels (e.g., "untranslatable" vs "translatable")

    Returns:
        Dict mapping group_label -> mean silhouette score
    """
    results = {}
    unique_groups = np.unique(group_labels)

    for group in unique_groups:
        mask = group_labels == group
        n = mask.sum()
        n_labels = len(np.unique(labels[mask]))
        if n < 2 or n_labels < 2 or n_labels >= n:
            continue

        sil = silhouette_score(activations[mask], labels[mask])
        results[group] = sil

    return results


def cluster_per_layer(
    hidden_states: Dict[int, np.ndarray],
    true_labels: np.ndarray,
    n_clusters: int = 10,
    n_components: int = 50,
    random_state: int = 42,
) -> Dict[int, Dict[str, float]]:
    """
    Run clustering analysis at each layer to find which layers show best separation.

    Args:
        hidden_states: Dict mapping layer_idx -> [n_prompts, d_model]
        true_labels: True language labels
        n_clusters: Number of k-means clusters
        n_components: PCA dimensions
        random_state: Random seed

    Returns:
        Dict mapping layer_idx -> metrics
    """
    results = {}
    for layer_idx in sorted(hidden_states.keys()):
        hs = hidden_states[layer_idx]

        # PCA reduction
        pca = PCA(n_components=min(n_components, hs.shape[1]), random_state=random_state)
        hs_pca = pca.fit_transform(hs)

        # Cluster
        pred_labels = cluster_kmeans(hs_pca, n_clusters=n_clusters, random_state=random_state)

        # Evaluate
        metrics = evaluate_clustering(pred_labels, true_labels, hs_pca)
        results[layer_idx] = metrics

        print(f"  Layer {layer_idx:2d}: ARI={metrics['adjusted_rand_index']:.4f}, "
              f"silhouette={metrics['silhouette_score']:.4f}")

    return results
