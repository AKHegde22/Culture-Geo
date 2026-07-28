#!/usr/bin/env python3
"""
Step 2: Extract activations from Llama-3-8B.

Runs the model on all prompts and captures:
    - Residual stream at each layer (hidden_states)
    - Final logits (for logit lens)

Supports both local GPU and Modal serverless execution.

Usage:
    # Local GPU
    python scripts/02_extract_activations.py --model meta-llama/Meta-Llama-3-8B

    # Modal serverless (L4 GPU, $0.80/hr)
    python scripts/02_extract_activations.py --modal --gpu L4

    # Specific layers only (saves memory)
    python scripts/02_extract_activations.py --layers 8,16,24,32
"""

import sys
import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_local(args):
    """Run extraction on local GPU."""
    import torch
    from src.extraction.extract import load_model, extract_activations, save_activations

    # Load prompts
    prompts_path = os.path.join(args.prompts_dir, "prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    print(f"Loaded {len(prompts)} prompts")

    # Parse layers
    layers = None
    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]

    # Load model
    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
    )

    # Extract
    start_time = time.time()
    results = extract_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        include_attention=args.include_attention,
        device=args.device,
        layers=layers,
    )
    elapsed = time.time() - start_time

    # Save
    save_activations(
        results=results,
        output_dir=args.output_dir,
        compress=args.compress,
    )

    print(f"\nExtraction complete in {elapsed:.1f}s")
    print(f"  Speed: {len(prompts) / elapsed:.1f} prompts/sec")


def run_modal(args):
    """Run extraction on Modal serverless GPU."""
    import subprocess
    import sys

    # Run the Modal app
    modal_script = PROJECT_ROOT / "src" / "extraction" / "modal_extract.py"

    cmd = [
        sys.executable, str(modal_script),
        "--prompts-file", os.path.join(args.prompts_dir, "prompts.json"),
        "--output-dir", args.output_dir,
        "--model-name", args.model,
        "--batch-size", str(args.batch_batch_size),
    ]

    print(f"Running Modal extraction: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract activations from LLM"
    )

    # Mode
    parser.add_argument("--modal", action="store_true",
                        help="Use Modal serverless GPU instead of local")

    # Model
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B",
                        help="HuggingFace model name")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16", "auto"],
                        help="Model dtype")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda, cpu, auto)")

    # Data
    parser.add_argument("--prompts-dir", default="data/processed",
                        help="Directory containing prompts.json")
    parser.add_argument("--output-dir", default="data/activations",
                        help="Output directory for activations")

    # Extraction
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for extraction")
    parser.add_argument("--max-length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all)")
    parser.add_argument("--include-attention", action="store_true",
                        help="Also extract attention patterns")
    parser.add_argument("--compress", action="store_true", default=True,
                        help="Compress output files")

    # Modal-specific
    parser.add_argument("--gpu", default="L4",
                        help="Modal GPU type (L4, A10G, A100)")
    parser.add_argument("--modal-batch-size", type=int, default=50,
                        help="Batch size for Modal extraction")

    args = parser.parse_args()

    print("=" * 60)
    print("Step 2: Extracting Activations")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Mode: {'Modal' if args.modal else 'Local'}")
    if not args.modal:
        print(f"  Device: {args.device}")
        print(f"  Batch size: {args.batch_size}")
    print()

    if args.modal:
        run_modal(args)
    else:
        run_local(args)

    print("\n" + "=" * 60)
    print("Done! Activations ready for analysis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
