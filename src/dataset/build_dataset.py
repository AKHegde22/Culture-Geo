"""
Build the complete dataset for the Untranslatability study.

Usage:
    python scripts/01_build_dataset.py
    python scripts/01_build_dataset.py --num-concepts-per-language 25 --seed 123
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.concepts import UNTRANSLATABLE_CONCEPTS, TRANSLATABLE_CONCEPTS, ALL_CONCEPTS
from src.dataset.prompts import generate_prompts, summarize_prompts, ALL_TEMPLATES


def build_dataset(
    output_dir: str = "data/processed",
    num_concepts_per_language: Optional[int] = None,
    seed: int = 42,
    cleaned: bool = False,
) -> dict:
    """
    Build the complete dataset.

    Args:
        output_dir: Directory to write output files
        num_concepts_per_language: If set, sample this many concepts per language
        seed: Random seed for reproducibility
        cleaned: If True, apply the frozen Path-B audit exclusions and write
            matching diagnostics.

    Returns:
        Dictionary with dataset statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    if cleaned:
        from src.dataset.audit import cleaned_concepts, matching_diagnostics

        concepts = cleaned_concepts()
    else:
        concepts = list(ALL_CONCEPTS)

    # Optionally subsample per language
    if num_concepts_per_language is not None:
        import random
        random.seed(seed)
        by_lang = {}
        for c in concepts:
            key = (c["language"], c["translatability"])
            by_lang.setdefault(key, []).append(c)

        sampled = []
        for key, group in by_lang.items():
            if len(group) > num_concepts_per_language:
                sampled.extend(random.sample(group, num_concepts_per_language))
            else:
                sampled.extend(group)
        concepts = sampled

    # Generate all prompts
    prompts = generate_prompts(concepts, seed=seed)
    stats = summarize_prompts(prompts)

    # Write concepts JSON
    concepts_path = os.path.join(output_dir, "concepts.json")
    with open(concepts_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2, ensure_ascii=False)

    # Write prompts JSON
    prompts_path = os.path.join(output_dir, "prompts.json")
    prompts_dicts = [
        {
            "text": p.text,
            "template_name": p.template_name,
            "concept_word": p.concept_word,
            "language": p.language,
            "language_code": p.language_code,
            "translatability": p.translatability,
            "category": p.category,
            "concept_index": p.concept_index,
        }
        for p in prompts
    ]
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts_dicts, f, indent=2, ensure_ascii=False)

    # Write manifest
    manifest = {
        "num_concepts": len(concepts),
        "num_prompts": len(prompts),
        "num_untranslatable": len([c for c in concepts if c["translatability"] == "untranslatable"]),
        "num_translatable": len([c for c in concepts if c["translatability"] == "translatable"]),
        "languages": list(set(c["language"] for c in concepts)),
        "templates": [t.name for t in ALL_TEMPLATES],
        "seed": seed,
        "cleaned": cleaned,
        "stats": stats,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if cleaned:
        from src.dataset.audit import matching_diagnostics

        diagnostics = matching_diagnostics(concepts)
        diag_path = os.path.join(output_dir, "matching_diagnostics.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False)
        manifest["matching_diagnostics_path"] = diag_path
        print(f"  Matching diagnostics: {diag_path}")
        print(
            f"  Excluded: {diagnostics['n_excluded']} | "
            f"same-category match frac: "
            f"{diagnostics['control_matching']['same_category_fraction']:.2f}"
        )

    print(f"Dataset built successfully:")
    print(f"  Concepts: {manifest['num_concepts']}")
    print(f"  Prompts:  {manifest['num_prompts']}")
    print(f"  Untranslatable: {manifest['num_untranslatable']}")
    print(f"  Translatable (control): {manifest['num_translatable']}")
    print(f"  Languages: {', '.join(manifest['languages'])}")
    print(f"\n  Output: {output_dir}/")
    print(f"    - concepts.json ({os.path.getsize(concepts_path) / 1024:.1f} KB)")
    print(f"    - prompts.json ({os.path.getsize(prompts_path) / 1024:.1f} KB)")
    print(f"    - manifest.json")

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the untranslatability dataset")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--num-concepts-per-language", type=int, default=None,
                        help="Sample N concepts per language (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--cleaned",
        action="store_true",
        help="Apply Path-B audit exclusions and write matching diagnostics",
    )
    args = parser.parse_args()

    build_dataset(
        output_dir=args.output_dir,
        num_concepts_per_language=args.num_concepts_per_language,
        seed=args.seed,
        cleaned=args.cleaned,
    )
