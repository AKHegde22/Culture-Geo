#!/usr/bin/env python3
"""
Step 4: Generate all paper figures.

Reads analysis results and generates publication-quality figures.

Usage:
    python scripts/04_generate_figures.py
    python scripts/04_generate_figures.py --results-dir data/analysis --output-dir paper/figures
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.figures import generate_all_figures


def main():
    import argparse
    from src.extraction.model_config import resolve_model_keys

    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys or 'all' (default: all)")
    parser.add_argument("--results-dir", default="data/analysis")
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--activations-dir", default="data/activations")
    args = parser.parse_args()

    model_keys = resolve_model_keys(args.models)

    print("=" * 60)
    print("Step 4: Generating Figures")
    print("=" * 60)
    print(f"  Models: {model_keys}")
    print()

    generate_all_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        activations_dir=args.activations_dir,
        model_keys=model_keys,
    )

    print("\n" + "=" * 60)
    print("Done! Figures ready for paper.")
    print("=" * 60)


if __name__ == "__main__":
    main()
