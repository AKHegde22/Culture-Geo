#!/usr/bin/env python3
"""
Step 7: Stratified per-prompt English-probability analysis (Step 2 of the
measurement fixes). Stratifies by template and language and fits a
mixed-effects model with template + language fixed effects.

Usage:
    python scripts/07_stratified_analysis.py --models all
"""

import argparse
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

    from src.analysis.stratified import run_stratified_analysis

    for mk in resolve_model_keys(args.models):
        cfg = get_model_config(mk)
        print("=" * 70)
        print(f"Stratified analysis: {mk} ({cfg['short_name']})")
        print("=" * 70)

        act_dir = os.path.join(args.activations_dir, mk)
        out_dir = os.path.join(args.output_dir, mk)
        os.makedirs(out_dir, exist_ok=True)

        results = run_stratified_analysis(act_dir, out_dir, cfg["hf_name"])
        save_results(results, os.path.join(out_dir, "stratified.json"))

        print("\n  Mixed model (template+language fixed effects):")
        for k, v in results["mixed_model"].items():
            print(f"    {k}: coef={v['coef']:.4f} p={v['p_value']:.4f}")
        print("\n  Per-language mid-layer Welch (untrans vs trans):")
        for lang, r in sorted(results["by_language"].items()):
            w = r["welch_midlayer"]
            print(f"    {lang:3s} n={r['n_untrans']}/{r['n_trans']:<4d} "
                  f"d={w['cohens_d']:+.3f} p={w['p_value']:.4f}")


if __name__ == "__main__":
    main()
