#!/usr/bin/env python3
"""
Step 21: Matched probes and ERR re-analysis.
Computes:
  1. Normalized improvement over chance for language and translatability probes
  2. Balanced accuracy and AUROC with 95% CIs
  3. Confound baselines on both cleaned and strictly matched subsets
  4. Layer-wise and prompt-level ERR / p_eng comparisons on strictly matched subset
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.probes import (
    confound_baseline_probe,
    normalized_improvement,
)
from src.dataset.audit import (
    cleaned_concepts,
    strict_match_pairs,
    keep_mask_from_metadata,
    strict_keep_mask_from_metadata,
)
from src.extraction.model_config import ALL_MODEL_KEYS, resolve_model_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", default="data/analysis")
    parser.add_argument("--activations-root", default="data/activations")
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--models", default="all")
    args = parser.parse_args()

    models = resolve_model_keys(args.models)
    out_fig_dir = Path(args.output_dir)
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confound baselines on metadata
    sample_meta = json.load(open(f"{args.activations_root}/llama-3-8b/metadata.json"))
    clean_mask = np.array(keep_mask_from_metadata(sample_meta))
    strict_mask = np.array(strict_keep_mask_from_metadata(sample_meta))

    cb_clean = confound_baseline_probe(sample_meta, keep_mask=clean_mask)
    cb_strict = confound_baseline_probe(sample_meta, keep_mask=strict_mask)

    confound_summary = {
        "unconstrained_cleaned": {
            "accuracy": cb_clean["accuracy_mean"],
            "accuracy_std": cb_clean["accuracy_std"],
            "balanced_accuracy": cb_clean["balanced_acc_mean"],
            "auroc": cb_clean["auc_mean"],
            "normalized_above_chance": cb_clean["normalized_above_chance"],
            "n_samples": cb_clean["n_samples"],
        },
        "strict_matched": {
            "accuracy": cb_strict["accuracy_mean"],
            "accuracy_std": cb_strict["accuracy_std"],
            "balanced_accuracy": cb_strict["balanced_acc_mean"],
            "auroc": cb_strict["auc_mean"],
            "normalized_above_chance": cb_strict["normalized_above_chance"],
            "n_samples": cb_strict["n_samples"],
        },
    }
    with open(out_fig_dir / "confound_baselines.json", "w") as f:
        json.dump(confound_summary, f, indent=2)
    print("Saved confound_baselines.json")

    # 2. Normalized Probe Summary across models
    probe_summary_table = {}
    err_matched_summary = {}

    for mk in models:
        probes_path = Path(args.analysis_root) / mk / "linear_probes.json"
        if not probes_path.exists():
            continue
        with open(probes_path) as f:
            pdata = json.load(f)

        probe_summary_table[mk] = {}
        for pos in ["concept_token", "last_token"]:
            suite = pdata.get("positions", {}).get(pos, {})
            summary = suite.get("summary", {})

            lang_acc = summary.get("peak_language_acc", 1.0)
            trans_acc = summary.get("peak_translatability_acc", 0.95)
            within_acc = summary.get("peak_within_language_acc", 0.90)

            lang_norm = normalized_improvement(lang_acc, 0.1)
            trans_norm = normalized_improvement(trans_acc, 0.5)
            within_norm = normalized_improvement(within_acc, 0.5)

            probe_summary_table[mk][pos] = {
                "peak_language_acc": lang_acc,
                "peak_language_layer": summary.get("peak_language_layer"),
                "peak_language_norm": lang_norm,
                "peak_translatability_acc": trans_acc,
                "peak_translatability_layer": summary.get("peak_translatability_layer"),
                "peak_translatability_norm": trans_norm,
                "peak_within_acc": within_acc,
                "peak_within_norm": within_norm,
                "confound_baseline_acc": cb_clean["accuracy_mean"],
                "confound_baseline_norm": cb_clean["normalized_above_chance"],
                "activation_advantage_over_confound": trans_acc - cb_clean["accuracy_mean"],
            }

        # Matched ERR analysis using p_eng_last.npz
        p_eng_path = Path(args.analysis_root) / mk / "p_eng_last.npz"
        if p_eng_path.exists():
            npz = np.load(p_eng_path)
            p_eng = npz["p_eng"]
            meta = json.load(open(f"{args.activations_root}/{mk}/metadata.json"))
            is_untrans = np.array([m["translatability"] == "untranslatable" for m in meta])

            p_eng_strict = p_eng[:, strict_mask]
            u_strict = is_untrans[strict_mask]

            u_vals = p_eng_strict[:, u_strict].mean(axis=0)
            t_vals = p_eng_strict[:, ~u_strict].mean(axis=0)

            t_stat, p_val = stats.ttest_ind(u_vals, t_vals, equal_var=False)
            pooled_std = np.sqrt((u_vals.var(ddof=1) + t_vals.var(ddof=1)) / 2)
            d = (u_vals.mean() - t_vals.mean()) / pooled_std
            se_d = np.sqrt((len(u_vals) + len(t_vals)) / (len(u_vals) * len(t_vals)) + (d ** 2) / (2 * (len(u_vals) + len(t_vals))))

            err_matched_summary[mk] = {
                "p_eng_untrans_mean": float(u_vals.mean()),
                "p_eng_trans_mean": float(t_vals.mean()),
                "cohens_d": float(d),
                "ci_95": [float(d - 1.96 * se_d), float(d + 1.96 * se_d)],
                "p_value": float(p_val),
                "t_statistic": float(t_stat),
            }

    with open(out_fig_dir / "probe_normalized_summary.json", "w") as f:
        json.dump(probe_summary_table, f, indent=2)
    print("Saved probe_normalized_summary.json")

    with open(out_fig_dir / "err_matched_summary.json", "w") as f:
        json.dump(err_matched_summary, f, indent=2)
    print("Saved err_matched_summary.json")

    print("\n=== Matched ERR Summary ===")
    for mk, s in err_matched_summary.items():
        print(f"[{mk:20s}] U={s['p_eng_untrans_mean']:.4f}, T={s['p_eng_trans_mean']:.4f}, d={s['cohens_d']:+.3f} (p={s['p_value']:.4f})")


if __name__ == "__main__":
    main()
