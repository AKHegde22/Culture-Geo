"""
Main activation extraction pipeline.

Loads a HuggingFace model, processes all prompts, and saves activations.

Usage:
    # Local
    python scripts/02_extract_activations.py --model meta-llama/Meta-Llama-3-8B

    # With specific layers
    python scripts/02_extract_activations.py --layers 8,16,24,32
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.hooks import ActivationExtractor, ActivationCache


class PromptDataset(Dataset):
    """Dataset of tokenized prompts for batched extraction."""

    def __init__(self, prompts: List[Dict], tokenizer, max_length: int = 128):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        encoding = self.tokenizer(
            prompt["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "metadata": {
                "text": prompt["text"],
                "template_name": prompt["template_name"],
                "concept_word": prompt["concept_word"],
                "language": prompt["language"],
                "language_code": prompt["language_code"],
                "translatability": prompt["translatability"],
                "category": prompt["category"],
                "concept_index": prompt["concept_index"],
            }
        }


def load_model(
    model_name: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading model: {model_name}")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        output_hidden_states=False,
    )
    model.eval()

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    return model, tokenizer


def extract_activations(
    model,
    tokenizer,
    prompts: List[Dict],
    batch_size: int = 8,
    max_length: int = 128,
    include_attention: bool = False,
    device: str = "cuda",
    layers: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Extract activations for all prompts.

    Args:
        model: HuggingFace model
        tokenizer: Corresponding tokenizer
        prompts: List of prompt dictionaries
        batch_size: Batch size for extraction
        max_length: Maximum sequence length
        include_attention: Whether to extract attention weights
        device: Device to run on
        layers: If specified, only extract these layers (saves memory)

    Returns:
        List of dictionaries with activations and metadata
    """
    extractor = ActivationExtractor(
        model=model,
        include_attention=include_attention,
        device=device,
    )

    dataset = PromptDataset(prompts, tokenizer, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    results = []
    total_tokens = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting activations")):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        batch_metadata = batch["metadata"]

        batch_size_actual = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        total_tokens += batch_size_actual * seq_len

        # Extract activations
        cache = extractor.extract(
            input_ids=input_ids,
            attention_mask=attention_mask,
            capture_logits=True,
        )

        # Process each item in the batch
        for i in range(batch_size_actual):
            # Get the last non-padding position for each sequence
            mask = attention_mask[i]
            last_pos = mask.sum().item() - 1

            # Extract hidden states at the last real token position
            # This represents the model's "understanding" of the full prompt
            hidden_states = {}
            for layer_idx, hs in cache.hidden_states.items():
                # hs shape: [batch, seq_len, d_model]
                # Take the last real token's representation
                hidden_states[layer_idx] = hs[i, last_pos, :].float().cpu()

            # Extract logits at the last position
            logits = None
            if cache.lm_logits is not None:
                logits = cache.lm_logits[i, last_pos, :].float().cpu()

            # Filter to requested layers
            if layers is not None:
                hidden_states = {
                    k: v for k, v in hidden_states.items() if k in layers
                }

            result = {
                "metadata": batch_metadata[i] if isinstance(batch_metadata[i], dict) else {
                    "text": batch_metadata["text"][i] if isinstance(batch_metadata.get("text"), list) else batch_metadata.get("text", ""),
                    "template_name": batch_metadata["template_name"][i] if isinstance(batch_metadata.get("template_name"), list) else batch_metadata.get("template_name", ""),
                    "concept_word": batch_metadata["concept_word"][i] if isinstance(batch_metadata.get("concept_word"), list) else batch_metadata.get("concept_word", ""),
                    "language": batch_metadata["language"][i] if isinstance(batch_metadata.get("language"), list) else batch_metadata.get("language", ""),
                    "language_code": batch_metadata["language_code"][i] if isinstance(batch_metadata.get("language_code"), list) else batch_metadata.get("language_code", ""),
                    "translatability": batch_metadata["translatability"][i] if isinstance(batch_metadata.get("translatability"), list) else batch_metadata.get("translatability", ""),
                    "category": batch_metadata["category"][i] if isinstance(batch_metadata.get("category"), list) else batch_metadata.get("category", ""),
                    "concept_index": batch_metadata["concept_index"][i] if isinstance(batch_metadata.get("concept_index"), list) else batch_metadata.get("concept_index", ""),
                },
                "hidden_states": {k: v.numpy() for k, v in hidden_states.items()},
                "logits": logits.numpy() if logits is not None else None,
                "seq_length": int(mask.sum().item()),
            }
            results.append(result)

        if batch_idx % 50 == 0:
            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches "
                  f"({total_tokens} tokens)")

    print(f"  Total: {len(results)} prompts, {total_tokens} tokens")
    return results


def save_activations(
    results: List[Dict],
    output_dir: str,
    compress: bool = True,
):
    """Save extracted activations to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata separately (lightweight)
    metadata = [r["metadata"] for r in results]
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Collect all unique layer indices
    all_layers = set()
    for r in results:
        all_layers.update(r["hidden_states"].keys())
    all_layers = sorted(all_layers)

    # Stack activations by layer
    for layer_idx in all_layers:
        layer_activations = []
        for r in results:
            if layer_idx in r["hidden_states"]:
                layer_activations.append(r["hidden_states"][layer_idx])

        if layer_activations:
            stacked = np.stack(layer_activations, axis=0)  # [n_prompts, d_model]
            layer_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.npy")

            if compress:
                # Save as compressed numpy
                np.savez_compressed(
                    layer_path.replace(".npy", ".npz"),
                    activations=stacked,
                )
            else:
                np.save(layer_path, stacked)

    # Stack logits
    logits_list = [r["logits"] for r in results if r["logits"] is not None]
    if logits_list:
        logits_stacked = np.stack(logits_list, axis=0)
        logits_path = os.path.join(output_dir, "logits.npy")
        if compress:
            np.savez_compressed(
                logits_path.replace(".npy", ".npz"),
                logits=logits_stacked,
            )
        else:
            np.save(logits_path, logits_stacked)

    # Save summary
    summary = {
        "num_results": len(results),
        "layers": all_layers,
        "d_model": int(results[0]["hidden_states"][all_layers[0]].shape[-1]) if results else 0,
        "has_logits": any(r["logits"] is not None for r in results),
    }
    with open(os.path.join(output_dir, "extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Calculate disk usage
    total_size = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))

    print(f"  Saved to {output_dir}/")
    print(f"    - {len(results)} prompt activations")
    print(f"    - {len(all_layers)} layers")
    print(f"    - d_model = {summary['d_model']}")
    print(f"    - Disk usage: {total_size / (1024**2):.1f} MB")
