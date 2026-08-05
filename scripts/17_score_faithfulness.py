#!/usr/bin/env python3
"""
Re-score existing generations with the faithfulness rubric
(and optionally an external LLM judge response file).

Usage:
    python scripts/17_score_faithfulness.py
    python scripts/17_score_faithfulness.py --generations-root data/generations
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def rescore_file(path: Path) -> list:
    from src.analysis.faithfulness import score_generation

    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    scored = []
    for row in rows:
        scores = score_generation(
            prediction=row.get("generation", ""),
            concept_word=row["concept_word"],
            language=row["language"],
            meaning=row.get("meaning", ""),
            translatability=row["translatability"],
            llm_response=row.get("llm_judge_raw"),
        )
        out = dict(row)
        out["scores"] = scores
        scored.append(out)
    return scored


def main():
    from src.analysis.faithfulness import aggregate_faithfulness

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations-root", default="data/generations")
    args = parser.parse_args()

    root = Path(args.generations_root)
    if not root.exists():
        print(f"No generations at {root}")
        return

    cross = {}
    for model_dir in sorted(root.iterdir()):
        gen_path = model_dir / "generations.json"
        if not gen_path.exists():
            continue
        scored = rescore_file(gen_path)
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2, ensure_ascii=False)
        agg = aggregate_faithfulness(scored)
        with open(model_dir / "faithfulness_stats.json", "w") as f:
            json.dump(agg, f, indent=2)
        cross[model_dir.name] = agg
        print(f"[{model_dir.name}] {agg['overall_untrans_vs_trans']}")

    with open(root / "cross_model_faithfulness.json", "w") as f:
        json.dump(cross, f, indent=2)


if __name__ == "__main__":
    main()
