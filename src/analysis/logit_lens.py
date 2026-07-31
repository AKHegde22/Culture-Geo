"""
Logit Lens analysis: What does the model predict at each layer?

The logit lens technique decodes intermediate representations into token
probabilities, revealing what the model "thinks" at each layer before
producing the final output.

Key metric: English Routing Rate (ERR)
    - At each layer, what fraction of tokens map to English vs. source language?
    - Higher ERR in mid-layers = model forces concepts through English space
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def apply_final_norm(
    hidden: np.ndarray,
    gamma: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply RMSNorm (as done before the LM head) to hidden states.

    Args:
        hidden: [n_prompts, d_model] or [d_model] activations
        gamma: RMSNorm weight vector [d_model] (elementwise scale)
        eps: RMSNorm epsilon

    Returns:
        Normed activations (same shape as input)
    """
    if gamma is None:
        return hidden
    hidden = hidden.astype(np.float32)
    gamma = gamma.astype(np.float32)
    rms = np.sqrt(np.mean(hidden.astype(np.float32) ** 2, axis=-1, keepdims=True) + eps)
    return hidden / rms * gamma


def compute_logit_lens(
    hidden_states: Dict[int, np.ndarray],
    logits: np.ndarray,
    tokenizer,
    lm_head_weights: Optional[np.ndarray] = None,
    lm_head_bias: Optional[np.ndarray] = None,
    final_norm_gamma: Optional[np.ndarray] = None,
) -> Dict:
    """
    Apply the logit lens technique to intermediate representations.

    For each layer's residual stream, project through the unembedding matrix
    to get what tokens the model would predict at that layer.

    Args:
        hidden_states: Dict mapping layer_idx -> [n_prompts, d_model] activations
        logits: Final layer logits [n_prompts, vocab_size]
        tokenizer: HuggingFace tokenizer
        lm_head_weights: Unembedding matrix [d_model, vocab_size] (if not using hidden_states directly)
        lm_head_bias: Unembedding bias [vocab_size]

    Returns:
        Dictionary with layer-wise predictions and English routing metrics
    """
    n_prompts = logits.shape[0]
    vocab_size = logits.shape[1]

    # Get English token IDs for comparison
    english_tokens = _get_english_token_ids(tokenizer)

    layer_predictions = {}
    english_routing = {}

    for layer_idx, hs in hidden_states.items():
        # Project through final norm + unembedding (if weights provided)
        if lm_head_weights is not None:
            layer_logits = apply_final_norm(hs, final_norm_gamma) @ lm_head_weights
            if lm_head_bias is not None:
                layer_logits += lm_head_bias
        else:
            # If no unembedding weights, we can't project
            # Skip this layer or use the final logits as reference
            continue

        # Get top-k predictions at each position
        top_k = 5
        top_k_indices = np.argpartition(layer_logits, -top_k, axis=-1)[:, -top_k:]
        top_k_values = np.take_along_axis(layer_logits, top_k_indices, axis=-1)

        # Sort by logit value
        sort_idx = np.argsort(-top_k_values, axis=-1)
        top_k_indices = np.take_along_axis(top_k_indices, sort_idx, axis=-1)

        # Decode predictions
        predictions = []
        for i in range(n_prompts):
            pred_tokens = [tokenizer.decode([idx]) for idx in top_k_indices[i]]
            predictions.append(pred_tokens)

        layer_predictions[layer_idx] = predictions

        # Compute English Routing Rate
        err = _compute_english_routing_rate(
            top_k_indices, english_tokens, tokenizer
        )
        english_routing[layer_idx] = err

    return {
        "layer_predictions": layer_predictions,
        "english_routing": english_routing,
        "n_prompts": n_prompts,
    }


def _get_english_token_ids(tokenizer) -> set:
    """Get set of token IDs that are primarily English words."""
    english_ids = set()

    # Strip tokenizer space/word-piece markers:
    #   " "     - WordPiece/BPE leading space
    #   "\u0120" (Ġ) - BPE leading-space marker (Llama/Qwen/GPT-2 style)
    #   "\u2581" (▁) - SentencePiece leading-space marker
    #   "##"    - WordPiece continuation marker
    markers = [" ", "\u0120", "\u2581", "##"]

    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        if token.startswith("<|") or token.startswith("["):
            continue
        clean = token
        for m in markers:
            clean = clean.replace(m, "")
        if clean and all(c in "abcdefghijklmnopqrstuvwxyz'" for c in clean):
            english_ids.add(token_id)

    return english_ids


def _compute_english_routing_rate(
    top_k_indices: np.ndarray,
    english_token_ids: set,
    tokenizer,
) -> float:
    """
    Compute English Routing Rate: fraction of top-1 tokens that are English.

    Higher ERR = more routing through English representation space.
    """
    n_prompts = top_k_indices.shape[0]
    english_count = 0
    total_count = 0

    for i in range(n_prompts):
        top_1_id = top_k_indices[i, 0]  # First element is top-1 after sorting
        if top_1_id in english_token_ids:
            english_count += 1
        total_count += 1

    return english_count / max(total_count, 1)


def compute_err_by_language(
    lens_results: Dict,
    metadata: List[Dict],
) -> Dict[str, Dict[int, float]]:
    """
    Compute English Routing Rate broken down by language.

    Returns:
        Dict mapping language_code -> {layer_idx: ERR}
    """
    english_routing = lens_results["english_routing"]

    # Group metadata by language
    lang_indices = {}
    for i, m in enumerate(metadata):
        lang = m["language_code"]
        if lang not in lang_indices:
            lang_indices[lang] = []
        lang_indices[lang].append(i)

    # We need per-prompt ERR, not aggregate
    # This requires re-computing with per-prompt granularity
    # For now, return the aggregate per layer
    return english_routing


def compute_err_by_translatability(
    lens_results: Dict,
    metadata: List[Dict],
) -> Dict[str, Dict[int, float]]:
    """
    Compute English Routing Rate broken down by translatability.

    Returns:
        Dict mapping "untranslatable"/"translatable" -> {layer_idx: ERR}
    """
    # This is a placeholder - full implementation requires per-prompt ERR
    # See the full extraction in the analyze script
    return lens_results["english_routing"]


def aggregate_err_by_translatability(
    hidden_states: Dict[int, np.ndarray],
    logits: np.ndarray,
    metadata: List[Dict],
    tokenizer,
    lm_head_weights: np.ndarray,
    lm_head_bias: Optional[np.ndarray] = None,
    final_norm_gamma: Optional[np.ndarray] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Compute per-prompt ERR, then aggregate by translatability.

    Returns:
        Dict mapping translatability -> {layer_idx: mean_ERR}
    """
    english_tokens = _get_english_token_ids(tokenizer)
    n_prompts = logits.shape[0]
    groups = {"untranslatable": [], "translatable": []}
    for i, m in enumerate(metadata):
        groups[m["translatability"]].append(i)

    results = {}
    for translatability, indices in groups.items():
        layer_errs = {}
        for layer_idx, hs in hidden_states.items():
            # Project through final norm + unembedding
            layer_logits = apply_final_norm(hs, final_norm_gamma) @ lm_head_weights
            if lm_head_bias is not None:
                layer_logits += lm_head_bias

            # Get top-1 for each prompt in this group
            top_1_indices = np.argmax(layer_logits[indices], axis=-1)

            # Count English tokens
            english_count = sum(
                1 for idx in top_1_indices if idx in english_tokens
            )
            err = english_count / len(indices)
            layer_errs[layer_idx] = err

        results[translatability] = layer_errs

    return results


def summarize_lens(lens_results: Dict) -> str:
    """Generate a text summary of logit lens results."""
    lines = ["Logit Lens Summary"]
    lines.append("=" * 40)

    for layer_idx in sorted(lens_results["english_routing"].keys()):
        err = lens_results["english_routing"][layer_idx]
        lines.append(f"  Layer {layer_idx:2d}: ERR = {err:.3f}")

    return "\n".join(lines)
