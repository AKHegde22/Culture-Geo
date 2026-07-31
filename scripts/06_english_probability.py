#!/usr/bin/env python3
"""
Step 6: Per-prompt graded English-probability analysis (Step 1 of the
measurement fixes). Computes softmax English probability mass per prompt per
layer, then runs properly powered statistical tests (per-prompt Welch test,
per-layer permutation tests, and a mixed-effects model).

Usage:
    python scripts/06_english_probability.py --models all
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.extraction.model_config import get_model_config, resolve_model_keys
from src.utils.io import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--activations-dir", default="data/activations")
    parser.add_argument("--output-dir", default="data/analysis")
    args = parser.parse_args()

    from src.analysis.english_probability import run_english_probability_analysis

    for mk in resolve_model_keys(args.models):
        cfg = get_model_config(mk)
        print("=" * 70)
        print(f"English probability analysis: {mk} ({cfg['short_name']})")
        print("=" * 70)

        act_dir = os.path.join(args.activations_dir, mk)
        out_dir = os.path.join(args.output_dir, mk)
        os.makedirs(out_dir, exist_ok=True)

        results = run_english_probability_analysis(act_dir, out_dir, cfg["hf_name"])
        save_results(results, os.path.join(out_dir, "english_probability.json"))

        w = results["welch_midlayer"]
        mm = results.get("mixed_model", {})
        inter = mm.get("untrans:depth", {})
        print(f"\n  Mid-layer per-prompt Welch: d={w['cohens_d']:.3f} "
              f"p={w['p_value']:.4f} (n={w['n_a']}/{w['n_b']})")
        if inter:
            print(f"  Mixed model untrans:depth: coef={inter['coef']:.4f} "
                  f"p={inter['p_value']:.4f}")
        else:
            print(f"  Mixed model: {mm}")


if __name__ == "__main__":
    main()
