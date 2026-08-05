#!/usr/bin/env python3
"""
Run linear probes on Modal using the culture-geo-activations Volume.

Usage:
    modal run scripts/15_modal_probes.py --models all
    modal run scripts/15_modal_probes.py --models llama-3-8b --layer-stride 2
"""

from __future__ import annotations

import json
import os
import sys

import modal

app = modal.App("culture-geo-probes")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "numpy",
        "scipy",
        "scikit-learn",
        "transformers",
        "torch",
        "huggingface_hub",
    )
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("scripts", remote_path="/root/scripts", copy=True)
)

act_volume = modal.Volume.from_name("culture-geo-activations", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": act_volume},
    cpu=8,
    memory=32768,
    timeout=3600,
    retries=0,
)
def run_probes(model_key: str, layer_stride: int = 1, cleaned: bool = True) -> dict:
    import importlib.util
    import shutil

    sys.path.insert(0, "/root")

    src = f"/data/{model_key}"
    local = f"/tmp/act/{model_key}"
    if not os.path.isdir(src):
        raise FileNotFoundError(f"No activations on volume at {src}")
    if os.path.exists(local):
        shutil.rmtree(local)
    shutil.copytree(src, local)

    spec = importlib.util.spec_from_file_location(
        "linear_probes", "/root/scripts/13_linear_probes.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out_dir = f"/tmp/results/{model_key}"
    os.makedirs(out_dir, exist_ok=True)
    return mod.run_for_model(
        model_key,
        local,
        out_dir,
        layer_stride=layer_stride,
        cleaned=cleaned,
        n_splits=5,
    )


@app.local_entrypoint()
def main(models: str = "all", layer_stride: int = 2, cleaned: bool = True):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.extraction.model_config import resolve_model_keys

    out_root = os.path.join(os.path.dirname(__file__), "..", "data", "analysis")
    cross = {}
    for mk in resolve_model_keys(models):
        print(f"[{mk}] running probes on Modal...")
        results = run_probes.remote(mk, layer_stride, cleaned)
        out_dir = os.path.join(out_root, mk)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "linear_probes.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {path}")
        cross[mk] = {
            pos: suite.get("summary", {})
            for pos, suite in results.get("positions", {}).items()
        }

    cross_path = os.path.join(out_root, "cross_model_probes.json")
    with open(cross_path, "w", encoding="utf-8") as f:
        json.dump(cross, f, indent=2)
    print(f"Wrote {cross_path}")
