"""
Utility functions for file I/O and data loading.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_activations(
    activation_dir: str,
    layers: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Load extracted activations from disk.

    Args:
        activation_dir: Directory containing layer_XX.npz files
        layers: If specified, only load these layers

    Returns:
        Dict mapping layer_idx -> [n_prompts, d_model] activations
    """
    activations = {}

    # Find all layer files
    layer_files = sorted(Path(activation_dir).glob("layer_*.npz"))

    for f in layer_files:
        # Extract layer index from filename
        layer_idx = int(f.stem.split("_")[-1])

        if layers is not None and layer_idx not in layers:
            continue

        data = np.load(f)
        activations[layer_idx] = data["activations"]

    if not activations:
        # Try .npy files as fallback
        layer_files = sorted(Path(activation_dir).glob("layer_*.npy"))
        for f in layer_files:
            layer_idx = int(f.stem.split("_")[-1])
            if layers is not None and layer_idx not in layers:
                continue
            activations[layer_idx] = np.load(f)

    print(f"  Loaded {len(activations)} layers from {activation_dir}")
    if activations:
        sample_layer = list(activations.keys())[0]
        print(f"  Shape per layer: {activations[sample_layer].shape}")

    return activations


def load_concept_activations(
    activation_dir: str,
    layers: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Load concept-token activations (at the concept word's position).

    Args:
        activation_dir: Directory containing concept_layer_XX.npz files
        layers: If specified, only load these layers

    Returns:
        Dict mapping layer_idx -> [n_prompts, d_model] activations
    """
    activations = {}
    layer_files = sorted(Path(activation_dir).glob("concept_layer_*.npz"))

    for f in layer_files:
        layer_idx = int(f.stem.replace("concept_layer_", ""))
        if layers is not None and layer_idx not in layers:
            continue
        data = np.load(f)
        activations[layer_idx] = data["activations"]

    if not activations:
        layer_files = sorted(Path(activation_dir).glob("concept_layer_*.npy"))
        for f in layer_files:
            layer_idx = int(f.stem.replace("concept_layer_", ""))
            if layers is not None and layer_idx not in layers:
                continue
            activations[layer_idx] = np.load(f)

    print(f"  Loaded {len(activations)} concept layers from {activation_dir}")
    if activations:
        sample_layer = list(activations.keys())[0]
        print(f"  Shape per concept layer: {activations[sample_layer].shape}")

    return activations


def load_logits(activation_dir: str) -> Optional[np.ndarray]:
    """Load extracted logits from disk."""
    logits_path = os.path.join(activation_dir, "logits.npz")
    if os.path.exists(logits_path):
        data = np.load(logits_path)
        return data["logits"]

    logits_path = os.path.join(activation_dir, "logits.npy")
    if os.path.exists(logits_path):
        return np.load(logits_path)

    return None


def load_metadata(activation_dir: str) -> List[Dict]:
    """Load prompt metadata from disk."""
    metadata_path = os.path.join(activation_dir, "metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lm_head(model_name: str, device: str = "cpu"):
    """
    Load the unembedding (LM head) matrix from a HuggingFace model.

    Returns:
        (weights, bias) tuple, or (None, None) if not available
    """
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu",
        )

        # Extract LM head weights (PyTorch shape [vocab, d_model] -> [d_model, vocab])
        if hasattr(model, "lm_head"):
            weights = model.lm_head.weight.data.cpu().numpy().T
            bias = None
            if model.lm_head.bias is not None:
                bias = model.lm_head.bias.data.cpu().numpy()
            del model
            return weights, bias

        del model
    except Exception as e:
        print(f"  Warning: Could not load LM head: {e}")

    return None, None


def load_lm_head_from_file(path: str):
    """
    Load LM head (unembedding) weights from a pre-saved .npz file.

    Returns:
        (weights, bias) tuple, or (None, None) if not available
    """
    if not os.path.exists(path):
        return None, None

    data = np.load(path, allow_pickle=True)
    weights = data["weights"]
    bias = None
    if "bias" in data.files:
        raw = data["bias"]
        if isinstance(raw, np.ndarray) and raw.ndim > 0 and raw.dtype.kind != "O":
            bias = raw
    if weights.ndim == 2:
        weights = weights.T
    return weights, bias


def load_final_norm_from_file(path: str):
    """
    Load final RMSNorm gamma weights from a pre-saved .npz file.

    Returns:
        gamma array (d_model,), or None if not available
    """
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    return data["gamma"]


def save_results(results: Dict, output_path: str):
    """Save analysis results to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"  Results saved to {output_path}")
