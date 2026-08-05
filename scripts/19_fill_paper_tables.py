#!/usr/bin/env python3
"""Fill paper Table faithfulness numbers from faithfulness_effect_table.json."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DISPLAY = {
    "llama-3-8b": "Llama-3-8B",
    "llama-3-8b-instruct": "Llama-3-8B-Instruct",
    "mistral-7b": "Mistral-7B",
    "qwen-2.5-7b": "Qwen2.5-7B",
}


def main():
    table_path = ROOT / "paper" / "figures" / "faithfulness_effect_table.json"
    tex_path = ROOT / "paper" / "main.tex"

    if not table_path.exists():
        print(f"Missing {table_path}; run generation + 18_pathb_figures.py first")
        return 1

    table = json.loads(table_path.read_text())
    order = list(DISPLAY.keys())
    rows = []
    for mk in order:
        s = table.get(mk, {})
        name = DISPLAY[mk]
        mu = s.get("mean_untrans")
        mt = s.get("mean_trans")
        d = s.get("cohens_d")
        if mu is None:
            rows.append(f"{name} & --- & --- & --- \\\\")
        else:
            rows.append(f"{name} & {mu:.3f} & {mt:.3f} & ${d:+.2f}$ \\\\")

    tex = tex_path.read_text()
    pattern = (
        r"(\\begin\{tabular\}\{lccc\}.*?\\midrule\n)"
        r"(.*?)"
        r"(\\bottomrule\n\\end\{tabular\}\n\\caption\{Generation faithfulness)"
    )
    m = re.search(pattern, tex, flags=re.S)
    if not m:
        print("Could not locate faithfulness table in main.tex (already filled?)")
        return 0

    body = "\n".join(rows) + "\n"
    new_tex = tex[: m.start(2)] + body + tex[m.end(2) :]
    tex_path.write_text(new_tex)
    print(f"Updated {tex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
