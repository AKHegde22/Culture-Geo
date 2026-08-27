#!/usr/bin/env python3
"""
Step 22: Generate all final paper figures incorporating:
  1. Updated condition-blind faithfulness bars and forest plot (with 95% CIs)
  2. Confound baseline vs matched subset plot
  3. Paired concrete examples export (JSON and LaTeX table fragment)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.plots import (
    plot_faithfulness_by_group,
    plot_faithfulness_effect_sizes,
    plot_confound_comparison,
)
from src.extraction.model_config import ALL_MODEL_KEYS


def main():
    out_dir = PROJECT_ROOT / "paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Faithfulness plots (Matched subset)
    matched_faith_path = PROJECT_ROOT / "data" / "generations" / "cross_model_faithfulness_matched.json"
    if matched_faith_path.exists():
        with open(matched_faith_path) as f:
            matched_faith = json.load(f)

        plot_faithfulness_by_group(
            matched_faith,
            output_path=str(out_dir / "cross_model_faithfulness.pdf"),
            title="Faithfulness on strictly matched pairs (mean ± 95% CI)",
        )
        plot_faithfulness_effect_sizes(
            matched_faith,
            output_path=str(out_dir / "cross_model_faithfulness_d.pdf"),
            title="Faithfulness effect size on matched pairs (Cohen's d with 95% CI)",
        )
        print("Updated faithfulness figures on matched subset.")

    # 2. Confound comparison plot
    confound_path = out_dir / "confound_baselines.json"
    if confound_path.exists():
        with open(confound_path) as f:
            cdata = json.load(f)
        plot_confound_comparison(
            cdata,
            output_path=str(out_dir / "confound_baseline_comparison.pdf"),
            title="Surface Confound Baseline: Unconstrained vs Strict Matched",
        )
        print("Generated confound_baseline_comparison.pdf")

    # 3. Extract 5 concrete paired examples across languages
    gen_path = PROJECT_ROOT / "data" / "generations" / "llama-3-8b-instruct" / "generations.json"
    if gen_path.exists():
        with open(gen_path) as f:
            generations = json.load(f)

        by_word = {}
        for d in generations:
            if d["template_name"] == "definition":
                by_word[d["concept_word"]] = d

        pairs_to_extract = [
            ("Tsundoku", "Tempura", "Japanese", "habit"),
            ("Schadenfreude", "Wanderlust", "German", "emotion"),
            ("Hygge", "Bakken", "Danish", "social"),
            ("Flaneur", "Restaurant", "French", "social"),
            ("Sobremesa", "Carnaval", "Portuguese", "social"),
        ]

        examples = []
        for u, t, lang, cat in pairs_to_extract:
            du = by_word.get(u)
            dt = by_word.get(t)
            if du and dt:
                examples.append({
                    "language": lang,
                    "category": cat,
                    "untranslatable": {
                        "word": u,
                        "meaning": du["meaning"],
                        "generation": du["generation"][:240],
                        "score": round(du["scores"]["overall"], 3),
                        "semantic": round(du["scores"]["semantic_correctness"], 3),
                        "nuance": round(du["scores"]["nuance_retained"], 3),
                        "cultural": round(du["scores"]["cultural_adequacy"], 3),
                    },
                    "translatable": {
                        "word": t,
                        "meaning": dt["meaning"],
                        "generation": dt["generation"][:240],
                        "score": round(dt["scores"]["overall"], 3),
                        "semantic": round(dt["scores"]["semantic_correctness"], 3),
                        "nuance": round(dt["scores"]["nuance_retained"], 3),
                        "cultural": round(dt["scores"]["cultural_adequacy"], 3),
                    },
                })

        with open(out_dir / "concrete_paired_examples.json", "w") as f:
            json.dump(examples, f, indent=2)
        print(f"Saved {len(examples)} paired examples to concrete_paired_examples.json")


if __name__ == "__main__":
    main()
