#!/usr/bin/env python3
"""
Generate Path-B figures: probes, faithfulness, concept-vs-last ARI.

Usage:
    python scripts/18_pathb_figures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    from src.extraction.model_config import ALL_MODEL_KEYS, resolve_model_keys
    from src.visualization.plots import (
        plot_concept_vs_last_ari,
        plot_faithfulness_by_group,
        plot_faithfulness_effect_sizes,
        plot_probe_accuracy_curves,
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--analysis-root", default="data/analysis")
    parser.add_argument("--generations-root", default="data/generations")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = resolve_model_keys(args.models)

    # Probe figures
    for mk in models:
        probes = _load(Path(args.analysis_root) / mk / "linear_probes.json")
        if not probes:
            print(f"[{mk}] no linear_probes.json — skip probe figs")
            continue
        for pos in ("last_token", "concept_token"):
            suite = probes.get("positions", {}).get(pos)
            if not suite:
                continue
            safe = mk.replace(".", "").replace("_", "-")
            out = os.path.join(args.output_dir, f"{safe}_probe_{pos}.pdf")
            plot_probe_accuracy_curves(
                suite["language_probe"],
                suite["translatability_probe"],
                output_path=out,
                title=f"{mk} ({pos}): language vs translatability probes",
                chance_language=suite.get("chance_language") or 0.1,
            )
            plt_close()

    # Concept vs last ARI overlays
    for mk in models:
        analysis = _load(Path(args.analysis_root) / mk / "analysis_results.json")
        concept = _load(Path(args.analysis_root) / mk / "concept_analysis.json")
        if not analysis or not concept:
            print(f"[{mk}] missing ARI sources — skip")
            continue
        ari_last = analysis.get("clustering", {}).get("ari_by_layer", {})
        ari_concept = concept.get("clustering", {}).get("ari_by_layer", {})
        if not ari_last or not ari_concept:
            print(f"[{mk}] incomplete ARI — skip")
            continue
        safe = mk.replace(".", "").replace("_", "-")
        out = os.path.join(args.output_dir, f"{safe}_ari_concept_vs_last.pdf")
        plot_concept_vs_last_ari(
            ari_last,
            ari_concept,
            output_path=out,
            title=f"{mk}: language ARI at concept vs last token",
        )
        plt_close()

    # Faithfulness figures
    faith = _load(Path(args.generations_root) / "cross_model_faithfulness.json")
    if not faith:
        # assemble from per-model files
        faith = {}
        for mk in models:
            s = _load(Path(args.generations_root) / mk / "faithfulness_stats.json")
            if s:
                faith[mk] = s
    if faith:
        plot_faithfulness_by_group(
            faith,
            output_path=os.path.join(args.output_dir, "cross_model_faithfulness.pdf"),
        )
        plt_close()
        plot_faithfulness_effect_sizes(
            faith,
            output_path=os.path.join(args.output_dir, "cross_model_faithfulness_d.pdf"),
        )
        plt_close()
        # Effect-size table JSON for the paper
        table = {}
        for mk, stats in faith.items():
            overall = stats.get("overall_untrans_vs_trans", {})
            table[mk] = {
                "cohens_d": overall.get("cohens_d"),
                "p_value": overall.get("p_value"),
                "mean_untrans": overall.get("mean_a"),
                "mean_trans": overall.get("mean_b"),
                "by_template": {
                    t: {
                        "cohens_d": v.get("cohens_d"),
                        "p_value": v.get("p_value"),
                    }
                    for t, v in stats.get("by_template", {}).items()
                },
            }
        with open(os.path.join(args.output_dir, "faithfulness_effect_table.json"), "w") as f:
            json.dump(table, f, indent=2)
        print("  Saved faithfulness_effect_table.json")
    else:
        print("No faithfulness stats yet — skip faithfulness figures")

    # Matching diagnostics summary for paper
    match = _load(PROJECT_ROOT / "data" / "processed_v2" / "matching_diagnostics.json")
    if match:
        summary = {
            "n_concepts": match.get("n_concepts"),
            "n_untranslatable": match.get("n_untranslatable"),
            "n_translatable": match.get("n_translatable"),
            "n_excluded": match.get("n_excluded"),
            "same_category_fraction": match.get("control_matching", {}).get(
                "same_category_fraction"
            ),
            "char_length_summary": match.get("char_length_summary"),
            "exclusions": match.get("exclusions"),
        }
        with open(os.path.join(args.output_dir, "matching_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("  Saved matching_summary.json")

    print("Path-B figures done.")


def plt_close():
    import matplotlib.pyplot as plt

    plt.close("all")


if __name__ == "__main__":
    main()
