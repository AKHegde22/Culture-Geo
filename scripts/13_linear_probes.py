#!/usr/bin/env python3
"""
Step 13: Linear probes for language vs. translatability.

Runs on last-token and concept-token activations for one or more models.
Supports a cleaned-concept keep-mask from the Path-B audit.

Usage:
    python scripts/13_linear_probes.py --models all
    python scripts/13_linear_probes.py --models llama-3-8b --layer-stride 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_for_model(
    model_key: str,
    activations_dir: str,
    output_dir: str,
    layer_stride: int = 1,
    cleaned: bool = True,
    n_splits: int = 5,
) -> dict:
    from src.analysis.probes import run_probe_suite, summarize_probe_contrast
    from src.dataset.audit import cleaned_concepts, keep_mask_from_metadata
    from src.utils.io import load_activations, load_concept_activations, load_metadata, save_results

    metadata = load_metadata(activations_dir)
    if cleaned:
        keep = keep_mask_from_metadata(metadata)
    else:
        keep = [True] * len(metadata)

    results = {
        "model": model_key,
        "cleaned": cleaned,
        "n_keep": int(sum(keep)),
        "positions": {},
    }

    last = load_activations(activations_dir)
    if last:
        print(f"  [{model_key}] probing last-token ({len(last)} layers)...")
        suite = run_probe_suite(
            last, metadata, keep_mask=keep, layer_stride=layer_stride, n_splits=n_splits
        )
        results["positions"]["last_token"] = suite
        results["positions"]["last_token"]["summary"] = summarize_probe_contrast(suite)

    concept = load_concept_activations(activations_dir)
    if concept:
        print(f"  [{model_key}] probing concept-token ({len(concept)} layers)...")
        suite = run_probe_suite(
            concept, metadata, keep_mask=keep, layer_stride=layer_stride, n_splits=n_splits
        )
        results["positions"]["concept_token"] = suite
        results["positions"]["concept_token"]["summary"] = summarize_probe_contrast(suite)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "linear_probes.json")
    save_results(results, out_path)
    print(f"  Saved {out_path}")

    for pos, suite in results["positions"].items():
        s = suite.get("summary", {})
        print(
            f"  {pos}: lang={s.get('peak_language_acc')} "
            f"trans={s.get('peak_translatability_acc')} "
            f"delta={s.get('language_minus_translatability')}"
        )
    return results


def main():
    from src.extraction.model_config import resolve_model_keys

    parser = argparse.ArgumentParser(description="Linear probes: language vs translatability")
    parser.add_argument("--models", default="all")
    parser.add_argument("--activations-root", default="data/activations")
    parser.add_argument("--output-root", default="data/analysis")
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--no-cleaned", action="store_true")
    args = parser.parse_args()

    keys = resolve_model_keys(args.models)
    all_summaries = {}
    for mk in keys:
        act_dir = os.path.join(args.activations_root, mk)
        if not os.path.isdir(act_dir):
            print(f"[{mk}] skip: missing {act_dir}")
            continue
        out_dir = os.path.join(args.output_root, mk)
        res = run_for_model(
            mk,
            act_dir,
            out_dir,
            layer_stride=args.layer_stride,
            cleaned=not args.no_cleaned,
            n_splits=args.n_splits,
        )
        all_summaries[mk] = {
            pos: suite.get("summary", {})
            for pos, suite in res.get("positions", {}).items()
        }

    cross_path = os.path.join(args.output_root, "cross_model_probes.json")
    with open(cross_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Wrote {cross_path}")


if __name__ == "__main__":
    main()
