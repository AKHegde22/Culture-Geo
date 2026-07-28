"""
Modal-based activation extraction for serverless GPU compute.

Runs activation extraction on Modal's serverless GPUs.
Supports L4 ($0.80/hr), A10G ($1.10/hr), A100 ($2.10/hr).

Usage:
    python scripts/02_extract_activations.py --modal --gpu L4
    modal run src/extraction/modal_extract.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

import modal

# Create Modal app
app = modal.App("culture-geo-extraction")

# Define the container image
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.37.0",
        "accelerate>=0.25.0",
        "tokenizers>=0.15.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    )
    .apt_install("git")
)

# Mount the project code
volume = modal.Volume.from_local_dir(".", mount_path="/root/culture-geo")


@app.function(
    image=image,
    gpu=modal.gpu.L4(count=1),
    timeout=3600,
    volumes={"/root/culture-geo": volume},
)
def extract_batch(
    prompt_batch: List[Dict],
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    dtype: str = "bfloat16",
    max_length: int = 128,
    include_attention: bool = False,
) -> List[Dict]:
    """
    Extract activations for a batch of prompts on Modal GPU.

    This function runs inside a Modal container with GPU access.
    """
    import sys
    sys.path.insert(0, "/root/culture-geo")

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        output_hidden_states=False,
    )
    model.eval()

    # Register hooks
    hooks = []
    cache = {}

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")

    for idx, layer in enumerate(layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                cache[f"hidden_{layer_idx}"] = hidden.detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # Process prompts
    results = []
    for prompt in prompt_batch:
        encoding = tokenizer(
            prompt["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to("cuda")
        attention_mask = encoding["attention_mask"].to("cuda")

        cache.clear()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits.detach()

        # Get last real token position
        last_pos = attention_mask.sum().item() - 1

        # Collect activations
        hidden_states = {}
        for k, v in cache.items():
            layer_idx = int(k.split("_")[-1])
            hidden_states[layer_idx] = v[0, last_pos, :].float().cpu().numpy()

        results.append({
            "metadata": prompt,
            "hidden_states": hidden_states,
            "logits": logits[0, last_pos, :].float().cpu().numpy(),
            "seq_length": int(attention_mask.sum().item()),
        })

    # Cleanup
    for hook in hooks:
        hook.remove()
    del model
    torch.cuda.empty_cache()

    return results


@app.function(
    image=image,
    timeout=3600,
    volumes={"/root/culture-geo": volume},
)
def save_results(
    results: List[Dict],
    output_dir: str,
):
    """Save extraction results to the mounted volume."""
    import numpy as np
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata = [r["metadata"] for r in results]
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Collect and save layer activations
    all_layers = set()
    for r in results:
        all_layers.update(r["hidden_states"].keys())

    for layer_idx in sorted(all_layers):
        layer_acts = np.stack(
            [r["hidden_states"][layer_idx] for r in results if layer_idx in r["hidden_states"]],
            axis=0,
        )
        np.savez_compressed(
            os.path.join(output_dir, f"layer_{layer_idx:02d}.npz"),
            activations=layer_acts,
        )

    # Save logits
    logits_list = [r["logits"] for r in results if r["logits"] is not None]
    if logits_list:
        np.savez_compressed(
            os.path.join(output_dir, "logits.npz"),
            logits=np.stack(logits_list, axis=0),
        )

    print(f"Saved {len(results)} results to {output_dir}")


@app.local_entrypoint()
def main(
    prompts_file: str = "data/processed/prompts.json",
    output_dir: str = "data/activations",
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    batch_size: int = 50,
    max_length: int = 128,
):
    """
    Main entry point for Modal-based extraction.

    Reads prompts from JSON, processes in batches, saves to mounted volume.
    """
    import json

    # Read prompts
    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts")

    # Process in batches
    all_results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}...")

        results = extract_batch.remote(
            prompt_batch=batch,
            model_name=model_name,
            max_length=max_length,
        )
        all_results.extend(results)

    # Save all results
    save_results.remote(all_results, output_dir)
    print(f"Done! Extracted {len(all_results)} activations")
