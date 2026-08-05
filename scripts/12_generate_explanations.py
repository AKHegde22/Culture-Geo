#!/usr/bin/env python3
"""
Step 12 (local): Generate + score concept explanations.

Prefer Modal for multi-model runs:
    modal run scripts/16_modal_generation.py --models all

Local GPU:
    python scripts/12_generate_explanations.py --models llama-3-8b --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GENERATION_TEMPLATES = ("definition", "comparison")


def load_concepts(concepts_path: str) -> List[Dict]:
    with open(concepts_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gen_prompts(concepts: List[Dict]) -> List[Dict]:
    from src.dataset.prompts import COMPARISON_TEMPLATE, DEFINITION_TEMPLATE

    templates = {"definition": DEFINITION_TEMPLATE, "comparison": COMPARISON_TEMPLATE}
    rows = []
    for idx, c in enumerate(concepts):
        for name in GENERATION_TEMPLATES:
            text = templates[name].format(
                word=c["word"],
                language=c["language"],
                native_script=c.get("native_script", c["word"]),
            )
            rows.append(
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
    return rows


def generate_local(
    model_key: str,
    prompts: List[Dict],
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    batch_size: int = 4,
) -> List[Dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.analysis.faithfulness import score_generation
    from src.extraction.model_config import get_model_config

    cfg = get_model_config(model_key)
    hf_name = cfg["hf_name"]
    print(f"Loading {hf_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()

    outputs = []
    do_sample = temperature > 0
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [r["prompt"] for r in batch]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
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
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return outputs


def main():
    from src.analysis.faithfulness import aggregate_faithfulness
    from src.extraction.model_config import resolve_model_keys

    parser = argparse.ArgumentParser(description="Generate + score cultural explanations")
    parser.add_argument("--models", default="llama-3-8b")
    parser.add_argument("--concepts", default="data/processed_v2/concepts.json")
    parser.add_argument("--output-dir", default="data/generations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    concepts = load_concepts(args.concepts)
    prompts = build_gen_prompts(concepts)
    print(f"Concepts={len(concepts)} generation prompts={len(prompts)}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_agg = {}
    for mk in resolve_model_keys(args.models):
        outs = generate_local(
            mk,
            prompts,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
        model_dir = os.path.join(args.output_dir, mk)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "generations.json"), "w", encoding="utf-8") as f:
            json.dump(outs, f, indent=2, ensure_ascii=False)
        agg = aggregate_faithfulness(outs)
        with open(os.path.join(model_dir, "faithfulness_stats.json"), "w") as f:
            json.dump(agg, f, indent=2)
        all_agg[mk] = agg
        print(f"[{mk}] overall: {agg['overall_untrans_vs_trans']}")

    with open(os.path.join(args.output_dir, "cross_model_faithfulness.json"), "w") as f:
        json.dump(all_agg, f, indent=2)


if __name__ == "__main__":
    main()
