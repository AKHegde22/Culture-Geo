#!/usr/bin/env python3
"""
Step 20: Re-score existing generations with the condition-blind rubric,
and compute rigorous statistics on both the full set and strict matched subset.

Reports:
  - Per-model effect size (Cohen's d) with 95% confidence intervals
  - Welch's t-test p-value
  - Within-matched-pair permutation test (10,000 permutations)
  - Confound regression (controlling for word length, category, language, token count)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.faithfulness import (
    score_generation,
    aggregate_faithfulness,
    welch_group_test,
    paired_permutation_test,
    confound_regression,
)
from src.dataset.audit import (
    cleaned_concepts,
    strict_match_pairs,
    matching_balance_table,
)


def rescore_file(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    scored = []
    for row in rows:
        scores = score_generation(
            prediction=row.get("generation", ""),
            concept_word=row["concept_word"],
            language=row["language"],
            meaning=row.get("meaning", ""),
            translatability=row.get("translatability"),
            llm_response=row.get("llm_judge_raw"),
        )
        out = dict(row)
        out["scores"] = scores
        scored.append(out)
    return scored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations-root", default="data/generations")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    root = Path(args.generations_root)
    out_fig_dir = Path(args.output_dir)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"Error: {root} does not exist.")
        return

    # 1. Get strict matched pairs
    concepts = cleaned_concepts()
    pairs = strict_match_pairs(concepts)
    balance = matching_balance_table(concepts, pairs)

    print(f"Strict matched pairs: {len(pairs)} (100% category & script matched)")
    with open(out_fig_dir / "matching_balance_table.json", "w") as f:
        json.dump(balance, f, indent=2)

    matched_u_words = {p["untranslatable"] for p in pairs}
    matched_t_words = {p["translatable"] for p in pairs}

    cross_full = {}
    cross_matched = {}
    paper_table_rows = {}

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        gen_path = model_dir / "generations.json"
        if not gen_path.exists():
            continue

        model_key = model_dir.name
        scored = rescore_file(gen_path)

        # Save rescored generations
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2, ensure_ascii=False)

        # Full set aggregate
        agg_full = aggregate_faithfulness(scored)
        with open(model_dir / "faithfulness_stats.json", "w") as f:
            json.dump(agg_full, f, indent=2)
        cross_full[model_key] = agg_full

        # Matched subset filter
        matched_records = [
            r for r in scored
            if r["concept_word"] in matched_u_words or r["concept_word"] in matched_t_words
        ]
        agg_matched = aggregate_faithfulness(matched_records)

        # Paired permutation test on strictly matched pairs
        u_by_concept = {}
        t_by_concept = {}
        for r in matched_records:
            w = r["concept_word"]
            tmpl = r["template_name"]
            score = r["scores"]["overall"]
            if r["translatability"] == "untranslatable":
                u_by_concept[(w, tmpl)] = score
            else:
                t_by_concept[(w, tmpl)] = score

        u_paired, t_paired = [], []
        for p in pairs:
            u_w = p["untranslatable"]
            t_w = p["translatable"]
            for tmpl in ["definition", "comparison"]:
                if (u_w, tmpl) in u_by_concept and (t_w, tmpl) in t_by_concept:
                    u_paired.append(u_by_concept[(u_w, tmpl)])
                    t_paired.append(t_by_concept[(t_w, tmpl)])

        perm_res = paired_permutation_test(u_paired, t_paired)
        agg_matched["paired_permutation_test"] = perm_res

        with open(model_dir / "faithfulness_stats_matched.json", "w") as f:
            json.dump(agg_matched, f, indent=2)
        cross_matched[model_key] = agg_matched

        # Paper table summary row
        ov_m = agg_matched["overall_untrans_vs_trans"]
        reg_m = agg_matched["confound_regression"]
        paper_table_rows[model_key] = {
            "full": {
                "mean_untrans": agg_full["overall_untrans_vs_trans"]["mean_a"],
                "mean_trans": agg_full["overall_untrans_vs_trans"]["mean_b"],
                "cohens_d": agg_full["overall_untrans_vs_trans"]["cohens_d"],
                "ci_95": agg_full["overall_untrans_vs_trans"]["ci_95"],
                "welch_p": agg_full["overall_untrans_vs_trans"]["p_value"],
            },
            "matched": {
                "mean_untrans": ov_m["mean_a"],
                "mean_trans": ov_m["mean_b"],
                "cohens_d": ov_m["cohens_d"],
                "ci_95": ov_m["ci_95"],
                "welch_p": ov_m["p_value"],
                "perm_p": perm_res["p_value"],
                "n_pairs": perm_res["n_pairs"],
                "regression_coef": reg_m.get("is_untranslatable_coef"),
                "regression_pval": reg_m.get("is_untranslatable_pval"),
                "regression_ci": reg_m.get("ci_95"),
            },
        }

        print(
            f"[{model_key:20s}] Matched (N={perm_res['n_pairs']}): "
            f"Untrans={ov_m['mean_a']:.3f}, Trans={ov_m['mean_b']:.3f}, "
            f"d={ov_m['cohens_d']:+.3f} [CI: {ov_m['ci_95'][0]:+.2f}, {ov_m['ci_95'][1]:+.2f}], "
            f"Perm_p={perm_res['p_value']:.4f}, Reg_p={reg_m.get('is_untranslatable_pval', 1.0):.4f}"
        )

    with open(root / "cross_model_faithfulness.json", "w") as f:
        json.dump(cross_full, f, indent=2)
    with open(root / "cross_model_faithfulness_matched.json", "w") as f:
        json.dump(cross_matched, f, indent=2)

    effect_table_path = out_fig_dir / "faithfulness_effect_table.json"
    with open(effect_table_path, "w") as f:
        json.dump(paper_table_rows, f, indent=2)
    print(f"Saved {effect_table_path}")


if __name__ == "__main__":
    main()
