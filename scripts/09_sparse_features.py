#!/usr/bin/env python3
"""
Step 9: Sparse-feature analysis (Step 4 of the measurement fixes).

Learns a sparse dictionary of features at the mid layer of the concept-token
activations and tests whether sparse code structure differs between
untranslatable and translatable concepts.

Usage:
    python scripts/09_sparse_features.py --models all [--position concept|last]
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
    parser.add_argument("--position", default="concept", choices=["concept", "last"])
    parser.add_argument("--n-components", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (default: mid layer)")
    args = parser.parse_args()

    from src.analysis.sparse_features import run_sparse_analysis
    from src.utils.io import (
        load_concept_activations,
        load_activations,
        load_metadata,
    )

    for mk in resolve_model_keys(args.models):
        cfg = get_model_config(mk)
        print("=" * 70)
        print(f"Sparse-feature analysis: {mk} ({cfg['short_name']}) @ {args.position}")
        print("=" * 70)

        act_dir = os.path.join(args.activations_dir, mk)
        out_dir = os.path.join(args.output_dir, mk)
        os.makedirs(out_dir, exist_ok=True)

        if args.position == "concept":
            hidden_states = load_concept_activations(act_dir)
        else:
            hidden_states = load_activations(act_dir)
        metadata = load_metadata(act_dir)

        if args.layers:
            layers = [int(l) for l in args.layers.split(",")]
        else:
            mid = sorted(hidden_states.keys())[len(hidden_states) // 2]
            layers = [mid]

        all_results = {}
        for layer_idx in layers:
            print(f"  Layer {layer_idx}:")
            all_results[str(layer_idx)] = run_sparse_analysis(
                hidden_states, metadata, layer_idx,
                n_components=args.n_components, alpha=args.alpha,
            )
            c = all_results[str(layer_idx)]["classify"]
            fs = all_results[str(layer_idx)]["feature_stats"]
            print(f"    classify: bal_acc={c.get('balanced_accuracy', 0):.3f} "
                  f"perm_p={c.get('permutation_p', 1):.3f} | "
                  f"features |d|>0.5: {fs['n_features_d_abs_gt_0_5']}")

        save_results(all_results, os.path.join(out_dir, "sparse_features.json"))
        print(f"  Saved to {out_dir}/sparse_features.json")


if __name__ == "__main__":
    main()
