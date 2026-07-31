#!/usr/bin/env python3
"""
Step 11: Run the entire analysis pipeline (03, 06, 07, 08, 09, 10) inside
Modal GPU containers, so no heavy compute happens locally.

Flow:
    1. Client uploads a model's activations (layer_XX.npz, concept_layer_XX.npz,
       logits.npz, lm_head.npz, final_norm.npz, metadata.json, ...) to the
       "culture-geo-activations" Volume.
    2. A Modal L4 GPU function loads them, reruns every analysis, and returns
       a JSON bundle.
    3. The client writes the results to data/analysis/{key}/.

Usage:
    modal run scripts/11_modal_analysis.py --models all
    modal run scripts/11_modal_analysis.py --models llama-3-8b
"""

import argparse
import base64
import importlib.util
import json
import os
import sys

import modal

app = modal.App("culture-geo-analysis")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "statsmodels",
        "transformers",
        "torch",
        "huggingface_hub",
        "umap-learn",
    )
    .add_local_dir("src", remote_path="/src", copy=True)
    .add_local_dir("scripts", remote_path="/scripts", copy=True)
)

act_volume = modal.Volume.from_name("culture-geo-activations", create_if_missing=True)

ANALYSIS_FILES = (
    "metadata.json",
    "concept_positions.json",
    "extraction_summary.json",
    "logits.npz",
    "lm_head.npz",
    "final_norm.npz",
)
LAYER_PREFIXES = ("layer_", "concept_layer_")


def _load_script(path: str):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(f"_modal_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@app.function(
    image=image,
    volumes={"/data": act_volume},
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=5400,
    retries=0,
)
def run_all_analyses(model_key: str, hf_token: str, pca_components: int = 50,
                     n_clusters: int = 10) -> dict:
    import numpy as np

    for p in ["/", "/root"]:
        if p not in sys.path:
            sys.path.insert(0, p)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    from src.extraction.model_config import get_model_config
    from src.utils.io import (
        load_activations,
        load_concept_activations,
        load_metadata,
    )
    from transformers import AutoTokenizer

    hf_name = get_model_config(model_key)["hf_name"]
    act_dir = f"/data/{model_key}"
    tmp_out = f"/tmp/results/{model_key}"
    os.makedirs(tmp_out, exist_ok=True)

    import shutil

    local_act = f"/tmp/act/{model_key}"
    print(f"  copying {act_dir} -> {local_act} (volume seek workaround)...")
    shutil.copytree(act_dir, local_act, dirs_exist_ok=True)
    act_dir = local_act

    print(f"=== {model_key} ({hf_name}) ===")

    metadata = load_metadata(act_dir)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    from src.utils.io import load_lm_head_from_file, load_final_norm_from_file
    lm_head_weights, lm_head_bias = load_lm_head_from_file(
        os.path.join(act_dir, "lm_head.npz")
    )
    final_norm_gamma = load_final_norm_from_file(
        os.path.join(act_dir, "final_norm.npz")
    )

    bundle = {"model_key": model_key, "n_prompts": len(metadata)}

    # ── 03: main analysis (logit lens ERR, clustering, trajectory, stats) ──
    print("--- 03 main analysis ---")
    mod03 = _load_script("/scripts/03_run_analysis.py")
    mod03.run_analysis(
        activations_dir="/data",
        output_dir="/tmp/results",
        model_name=hf_name,
        pca_components=pca_components,
        n_clusters=n_clusters,
        model_key=model_key,
    )
    with open(os.path.join(tmp_out, "analysis_results.json")) as f:
        bundle["analysis_results.json"] = json.load(f)

    # ── 06: english probability (last position) ────────────────────────────
    print("--- 06 english probability ---")
    from src.analysis.english_probability import (
        compute_english_probability_mass,
        run_english_probability_analysis,
    )
    bundle["english_probability.json"] = run_english_probability_analysis(
        act_dir, tmp_out, hf_name, device="cuda"
    )

    # ── 07/10: stratified (last + concept) ─────────────────────────────────
    print("--- 07/10 stratified (last + concept) ---")
    from src.analysis.stratified import run_stratified_analysis
    bundle["stratified_last.json"] = run_stratified_analysis(
        act_dir, tmp_out, hf_name, position="last", device="cuda"
    )
    bundle["stratified_concept.json"] = run_stratified_analysis(
        act_dir, tmp_out, hf_name, position="concept", device="cuda"
    )

    # ── 08: concept-position analysis ──────────────────────────────────────
    print("--- 08 concept analysis ---")
    mod08 = _load_script("/scripts/08_concept_analysis.py")
    bundle["concept_analysis.json"] = mod08.run_concept_analysis(
        act_dir, tmp_out, hf_name, n_clusters, pca_components, device="cuda"
    )

    # ── 09: sparse features (last + concept, mid layer) ────────────────────
    print("--- 09 sparse features ---")
    from src.analysis.sparse_features import run_sparse_analysis
    for pos, loader in [("last", load_activations), ("concept", load_concept_activations)]:
        hs = loader(act_dir)
        layers = sorted(hs.keys())
        mid = layers[len(layers) // 2]
        bundle[f"sparse_features_{pos}.json"] = run_sparse_analysis(hs, metadata, mid)

    # ── Per-prompt English probability matrices (both positions) ───────────
    print("--- p_eng matrices ---")
    group_mask = np.array(
        [m["translatability"] == "untranslatable" for m in metadata], dtype=bool
    )
    bundle["p_eng_group_mask"] = [int(x) for x in group_mask.tolist()]
    for pos, loader in [("last", load_activations), ("concept", load_concept_activations)]:
        hs = loader(act_dir)
        p_eng, p_layers = compute_english_probability_mass(
            hs, tokenizer, lm_head_weights, lm_head_bias, final_norm_gamma,
            device="cuda",
        )
        bundle[f"p_eng_{pos}.b64"] = base64.b64encode(
            np.ascontiguousarray(p_eng).astype("<f4").tobytes()
        ).decode()
        bundle[f"p_eng_{pos}_layers"] = [int(l) for l in p_layers]

    print(f"=== {model_key} done ===")
    return bundle


def _upload_model_activations(model_key: str, activations_root: str) -> int:
    """Upload one model's activation files to the Volume (client-side)."""
    local_dir = os.path.join(activations_root, model_key)
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"{local_dir} not found")

    files = []
    for fname in sorted(os.listdir(local_dir)):
        if fname in ANALYSIS_FILES:
            files.append(fname)
        elif fname.startswith(LAYER_PREFIXES) and fname.endswith(".npz"):
            files.append(fname)
    if not files:
        raise FileNotFoundError(f"No activation files found in {local_dir}")

    print(f"  uploading {len(files)} files for {model_key}...")
    with act_volume.batch_upload(force=True) as batch:
        for fname in files:
            batch.put_file(os.path.join(local_dir, fname), f"/{model_key}/{fname}")
    return len(files)


@app.local_entrypoint()
def main(models: str = "all", pca_components: int = 50, n_clusters: int = 10):
    sys_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.insert(0, sys_path)
    from src.extraction.model_config import resolve_model_keys

    try:
        from huggingface_hub import get_token
        hf_token = get_token() or ""
    except Exception:
        hf_token = ""

    activations_root = os.path.join(sys_path, "data", "activations")
    out_root = os.path.join(sys_path, "data", "analysis")

    for mk in resolve_model_keys(models):
        print(f"[{mk}] upload activations to Volume...")
        n = _upload_model_activations(mk, activations_root)
        print(f"[{mk}] uploaded {n} files; dispatching GPU analysis...")

        bundle = run_all_analyses.remote(mk, hf_token, pca_components, n_clusters)
        out_dir = os.path.join(out_root, mk)
        os.makedirs(out_dir, exist_ok=True)

        for name in [
            "analysis_results.json",
            "english_probability.json",
            "stratified_last.json",
            "stratified_concept.json",
            "concept_analysis.json",
            "sparse_features_last.json",
            "sparse_features_concept.json",
        ]:
            if name in bundle:
                with open(os.path.join(out_dir, name), "w") as f:
                    json.dump(bundle[name], f, indent=2)
                print(f"  wrote {name}")

        import numpy as np
        for pos in ["last", "concept"]:
            b64 = bundle.get(f"p_eng_{pos}.b64")
            layers = bundle.get(f"p_eng_{pos}_layers")
            mask = bundle.get("p_eng_group_mask")
            if b64 and layers and mask:
                p = np.frombuffer(base64.b64decode(b64), dtype="<f4")
                p = p.reshape(len(layers), len(mask))
                np.savez_compressed(
                    os.path.join(out_dir, f"p_eng_{pos}.npz"),
                    p_eng=p,
                    layer_indices=np.array(layers),
                    group_mask=np.array(mask, dtype=bool),
                )
                print(f"  wrote p_eng_{pos}.npz ({p.shape})")

        print(f"Done {mk}")
    print("All models complete.")
