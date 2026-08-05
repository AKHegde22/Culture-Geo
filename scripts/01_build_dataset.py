#!/usr/bin/env python3
"""
Step 1: Build the untranslatability dataset.

Creates:
    - data/processed/concepts.json    (200+ cultural concepts)
    - data/processed/prompts.json     (600+ formatted prompts)
    - data/processed/manifest.json    (dataset statistics)

Usage:
    python scripts/01_build_dataset.py
    python scripts/01_build_dataset.py --num-concepts-per-language 15
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.build_dataset import build_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build the cultural untranslatability dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory (default: data/processed)"
    )
    parser.add_argument(
        "--num-concepts-per-language",
        type=int,
        default=None,
        help="Sample N concepts per language (default: use all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--cleaned",
        action="store_true",
        help="Apply Path-B audit exclusions and write matching diagnostics",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1: Building Dataset")
    print("=" * 60)

    manifest = build_dataset(
        output_dir=args.output_dir,
        num_concepts_per_language=args.num_concepts_per_language,
        seed=args.seed,
        cleaned=args.cleaned,
    )

    print("\n" + "=" * 60)
    print("Done! Dataset ready for extraction.")
    print("=" * 60)


if __name__ == "__main__":
    main()
