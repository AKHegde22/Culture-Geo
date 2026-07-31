"""
Modal-based activation extraction for serverless GPU compute.

Runs activation extraction on Modal's serverless GPUs (L4 $0.80/hr).
Extracts residual stream activations + final logits for every prompt,
and saves the LM head (unembedding) weights for logit-lens analysis.

Usage:
    python scripts/02_extract_activations.py --modal --models all
    python src/extraction/modal_extract.py \
        --prompts-file data/processed/prompts.json \
        --output-dir data/activations/llama-3-8b \
        --model-name meta-llama/Meta-Llama-3-8B \
        --model-key llama-3-8b
"""

import asyncio
import os
import sys
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
        "transformers>=4.46.0",
        "accelerate>=0.30.0",
        "tokenizers>=0.20.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
    )
    .apt_install("git")
)

# Volume used to store LM head (unembedding) weights per model
lm_volume = modal.Volume.from_name("culture-geo-lm-heads", create_if_missing=True)

# Persistent HF cache volume: avoid re-downloading multi-GB weights per invocation
hf_cache_volume = modal.Volume.from_name("culture-geo-hf-cache", create_if_missing=True)


def _find_concept_token_index(tokenizer, text: str, concept_word: str) -> int:
    """
    Find the token index of the LAST subword token covering the concept word.

    The concept appears as "Word" (double-quoted) in the prompt text. Returns
    the token index of the final token of the concept span, or -1 if the word
    cannot be located.
    """
    if not concept_word:
        return -1
    enc = tokenizer(
        text,
        max_length=128,
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = enc.get("offset_mapping")
    if offsets is None:
        return -1

    # Locate the quoted concept first, then the bare word
    quoted = f'"{concept_word}"'
    start = text.find(quoted)
    if start >= 0:
        start += 1  # skip the opening quote
    else:
        start = text.find(concept_word)
    if start < 0:
        return -1
    end = start + len(concept_word)

    selected = [
        i for i, (s, e) in enumerate(offsets)
        if s is not None and s < end and e > start
    ]
    if selected:
        return selected[-1]
    return -1


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    scaledown_window=600,
    volumes={"/lm": lm_volume, "/root/.cache/huggingface": hf_cache_volume},
)
def extract_batch(
    prompt_batch: List[Dict],
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    model_key: str = "llama-3-8b",
    hf_token: Optional[str] = None,
    max_length: int = 128,
    include_attention: bool = False,
    positions: str = "last,concept",
    include_logits: bool = True,
) -> List[Dict]:
    """
    Extract activations for a batch of prompts on Modal GPU.

    Runs a single batched forward pass per call and returns per-prompt
    hidden states (at the last real token position and/or the concept token
    position) and optionally final logits. Also writes the model's LM head
    weights to the /lm volume once.
    """
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        output_hidden_states=False,
        token=hf_token,
    )
    model.eval()

    # Register hooks on residual stream output of each layer
    hooks = []
    cache = {}

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")

    for idx, layer in enumerate(layers):
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    hidden = output
                else:
                    hidden = output[0]
                cache[f"hidden_{layer_idx}"] = hidden.detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    # Save LM head weights to volume (once per model)
    lm_head_path = f"/lm/{model_key}_lm_head.npz"
    if model_key and not os.path.exists(lm_head_path):
        try:
            if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
                w = model.lm_head.weight.data.float().cpu().numpy()
                b = None
                if model.lm_head.bias is not None:
                    b = model.lm_head.bias.data.float().cpu().numpy()
            elif hasattr(model.model, "embed_tokens"):
                w = model.model.embed_tokens.weight.data.float().cpu().numpy()
                b = None
            else:
                w, b = None, None
            if w is not None:
                save_kwargs = {"weights": w}
                if b is not None:
                    save_kwargs["bias"] = b
                np.savez_compressed(lm_head_path, **save_kwargs)
                lm_volume.commit()
                print(f"  Saved LM head to volume: {w.shape}")
            else:
                print("  WARNING: no LM head found to save")
        except Exception as e:
            print(f"  WARNING: could not save LM head: {e}")

    # Save final RMSNorm weights (applied before the LM head; needed for logit lens)
    final_norm_path = f"/lm/{model_key}_final_norm.npz"
    if model_key and not os.path.exists(final_norm_path):
        try:
            norm_mod = getattr(getattr(model, "model", None), "norm", None)
            if norm_mod is not None and hasattr(norm_mod, "weight"):
                gamma = norm_mod.weight.data.float().cpu().numpy()
                np.savez_compressed(final_norm_path, gamma=gamma)
                lm_volume.commit()
                print(f"  Saved final norm to volume: {gamma.shape}")
            else:
                print("  WARNING: no final norm found to save")
        except Exception as e:
            print(f"  WARNING: could not save final norm: {e}")

    # Tokenize the whole batch at once (pad to longest in batch)
    encodings = tokenizer(
        [p["text"] for p in prompt_batch],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"].to("cuda")
    attention_mask = encodings["attention_mask"].to("cuda")

    cache.clear()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits.detach()

    # Per-prompt extraction at requested positions
    want_last = "last" in positions.split(",")
    want_concept = "concept" in positions.split(",")

    results = []
    for i, prompt in enumerate(prompt_batch):
        seq_len = int(attention_mask[i].sum().item())

        hidden_states = {}
        concept_hidden_states = {}
        concept_token_index = -1

        if want_last:
            last_pos = seq_len - 1
            for k, v in cache.items():
                layer_idx = int(k.split("_")[-1])
                hidden_states[layer_idx] = v[i, last_pos, :].float().cpu().numpy()

        if want_concept:
            concept_token_index = _find_concept_token_index(
                tokenizer, prompt["text"], prompt.get("concept_word", "")
            )
            if 0 <= concept_token_index < seq_len:
                for k, v in cache.items():
                    layer_idx = int(k.split("_")[-1])
                    concept_hidden_states[layer_idx] = v[i, concept_token_index, :].float().cpu().numpy()

        result = {
            "metadata": prompt,
            "hidden_states": hidden_states,
            "concept_hidden_states": concept_hidden_states,
            "concept_token_index": concept_token_index,
            "seq_length": seq_len,
        }
        if include_logits:
            result["logits"] = logits[i, seq_len - 1, :].float().cpu().numpy()
        results.append(result)

    # Cleanup
    for hook in hooks:
        hook.remove()
    del model
    torch.cuda.empty_cache()

    return results


async def fetch_lm_head(model_key: str, output_dir: str):
    """Download the model's LM head weights from the Modal volume to local disk."""
    import numpy as np

    v = modal.Volume.from_name("culture-geo-lm-heads")
    fname = f"{model_key}_lm_head.npz"
    dest = os.path.join(output_dir, "lm_head.npz")

    for attempt in range(15):
        try:
            chunks = []
            for chunk in v.read_file(fname):
                chunks.append(chunk)
            data = b"".join(chunks)
            if data:
                with open(dest, "wb") as f:
                    f.write(data)
                with np.load(dest) as npz:
                    shape = npz["weights"].shape
                print(f"  LM head downloaded to {dest} (shape {shape})")
                break
        except Exception as e:
            print(f"  LM head fetch attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(10)
    else:
        print(f"  WARNING: could not fetch LM head for {model_key}")

    # Fetch final norm weights
    norm_fname = f"{model_key}_final_norm.npz"
    norm_dest = os.path.join(output_dir, "final_norm.npz")
    for attempt in range(15):
        try:
            chunks = []
            for chunk in v.read_file(norm_fname):
                chunks.append(chunk)
            data = b"".join(chunks)
            if data:
                with open(norm_dest, "wb") as f:
                    f.write(data)
                with np.load(norm_dest) as npz:
                    shape = npz["gamma"].shape
                print(f"  Final norm downloaded to {norm_dest} (shape {shape})")
                break
        except Exception as e:
            print(f"  Final norm fetch attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(10)
    else:
        print(f"  WARNING: could not fetch final norm for {model_key}")


@app.local_entrypoint()
async def main(
    prompts_file: str = "data/processed/prompts.json",
    output_dir: str = "data/activations",
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    model_key: str = "llama-3-8b",
    batch_size: int = 50,
    max_length: int = 128,
    positions: str = "last,concept",
    include_logits: str = "true",
):
    """
    Main entry point for Modal-based extraction.

    Reads prompts from JSON, processes in batches on Modal GPU, and saves
    activations + LM head weights to the local output directory.

    positions: comma-separated "last,concept" (which token positions to save).
    include_logits: "true"/"false"; whether to transfer/save final-layer logits.
    """
    import json
    from huggingface_hub import get_token

    include_logits = include_logits.lower() in ("true", "1", "yes")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    hf_token = get_token()

    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Model: {model_name} ({model_key})")
    print(f"Output: {output_dir}")
    print(f"Positions: {positions}, include_logits: {include_logits}")

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{n_batches} ({len(batch)} prompts)...")

        results = extract_batch.remote(
            prompt_batch=batch,
            model_name=model_name,
            model_key=model_key,
            hf_token=hf_token,
            max_length=max_length,
            positions=positions,
            include_logits=include_logits,
        )
        all_results.extend(results)

    # Save results locally (metadata.json, layer_XX.npz, concept_layer_XX.npz, logits.npz)
    from src.extraction.extract import save_activations

    save_activations(all_results, output_dir, compress=True)

    # Download LM head weights from the volume (needed for logit lens)
    await fetch_lm_head(model_key, output_dir)

    print(f"Done! Extracted {len(all_results)} activations to {output_dir}")
