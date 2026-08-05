#!/usr/bin/env python3
"""
Modal generation + faithfulness scoring for Path B.

Usage:
    modal run scripts/16_modal_generation.py --models all
    modal run scripts/16_modal_generation.py --models llama-3-8b
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("culture-geo-generation")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/root/src", copy=True)
    .add_local_file(
        str(PROJECT_ROOT / "data" / "processed_v2" / "concepts.json"),
        remote_path="/root/concepts.json",
        copy=True,
    )
)

hf_cache = modal.Volume.from_name("culture-geo-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    memory=32768,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def generate_model(model_key: str, hf_token: str, max_new_tokens: int = 128) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    sys.path.insert(0, "/root")

    from src.analysis.faithfulness import aggregate_faithfulness, score_generation
    from src.dataset.prompts import COMPARISON_TEMPLATE, DEFINITION_TEMPLATE
    from src.extraction.model_config import get_model_config

    with open("/root/concepts.json", "r", encoding="utf-8") as f:
        concepts = json.load(f)

    templates = {"definition": DEFINITION_TEMPLATE, "comparison": COMPARISON_TEMPLATE}
    prompts = []
    for idx, c in enumerate(concepts):
        for name in ("definition", "comparison"):
            text = templates[name].format(
                word=c["word"],
                language=c["language"],
                native_script=c.get("native_script", c["word"]),
            )
            prompts.append(
                {
                    "prompt": text,
                    "template_name": name,
                    "concept_word": c["word"],
                    "language": c["language"],
                    "language_code": c["language_code"],
                    "translatability": c["translatability"],
                    "category": c["category"],
                    "meaning": c.get("meaning", ""),
                    "concept_index": idx,
                }
            )

    cfg = get_model_config(model_key)
    hf_name = cfg["hf_name"]
    print(f"[{model_key}] loading {hf_name} ({len(prompts)} prompts)...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    # Decoder-only models need left padding for correct batched generation.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    outputs = []
    batch_size = 4
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [r["prompt"] for r in batch]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, row in enumerate(batch):
            prompt_len = int(enc["attention_mask"][j].sum().item())
            continuation = tokenizer.decode(
                gen[j][prompt_len:], skip_special_tokens=True
            ).strip()
            scores = score_generation(
                continuation,
                row["concept_word"],
                row["language"],
                row.get("meaning", ""),
                row["translatability"],
            )
            out = dict(row)
            out["generation"] = continuation
            out["model"] = model_key
            out["scores"] = scores
            outputs.append(out)
        if (i // batch_size) % 20 == 0:
            print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}")

    agg = aggregate_faithfulness(outputs)
    print(f"[{model_key}] overall d={agg['overall_untrans_vs_trans'].get('cohens_d')}")
    return {"generations": outputs, "faithfulness_stats": agg, "model": model_key}


@app.local_entrypoint()
def main(models: str = "all", max_new_tokens: int = 128):
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.extraction.model_config import resolve_model_keys

    try:
        from huggingface_hub import get_token

        hf_token = get_token() or ""
    except Exception:
        hf_token = os.environ.get("HF_TOKEN", "")

    out_root = PROJECT_ROOT / "data" / "generations"
    out_root.mkdir(parents=True, exist_ok=True)
    cross = {}

    for mk in resolve_model_keys(models):
        print(f"=== {mk} ===")
        bundle = generate_model.remote(mk, hf_token, max_new_tokens)
        model_dir = out_root / mk
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "generations.json", "w", encoding="utf-8") as f:
            json.dump(bundle["generations"], f, indent=2, ensure_ascii=False)
        with open(model_dir / "faithfulness_stats.json", "w") as f:
            json.dump(bundle["faithfulness_stats"], f, indent=2)
        cross[mk] = bundle["faithfulness_stats"]
        print(f"  wrote {model_dir}")

    with open(out_root / "cross_model_faithfulness.json", "w") as f:
        json.dump(cross, f, indent=2)
    print("Done.")
