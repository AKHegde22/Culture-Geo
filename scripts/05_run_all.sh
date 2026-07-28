#!/bin/bash
# ───────────────────────────────────────────────────────────────────────────
# Culture-Geo: Full Pipeline Runner
# ───────────────────────────────────────────────────────────────────────────
# Runs the complete experiment pipeline:
#   1. Build dataset
#   2. Extract activations
#   3. Run analysis
#   4. Generate figures
#
# Usage:
#   bash scripts/05_run_all.sh
#   bash scripts/05_run_all.sh --modal
#   bash scripts/05_run_all.sh --layers 8,16,24,32
# ───────────────────────────────────────────────────────────────────────────

set -e  # Exit on error

# Parse arguments
USE_MODAL=false
EXTRA_ARGS=""
LAYERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --modal)
            USE_MODAL=true
            shift
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "============================================================"
echo "Culture-Geo: Full Pipeline"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Mode: $([ "$USE_MODAL" = true ] && echo 'Modal Serverless' || echo 'Local GPU')"
echo "  Layers: ${LAYERS:-all}"
echo ""

# Step 1: Build Dataset
echo "------------------------------------------------------------"
echo "Step 1/4: Building Dataset"
echo "------------------------------------------------------------"
python scripts/01_build_dataset.py
echo ""

# Step 2: Extract Activations
echo "------------------------------------------------------------"
echo "Step 2/4: Extracting Activations"
echo "------------------------------------------------------------"
EXTRACT_ARGS=""
if [ "$USE_MODAL" = true ]; then
    EXTRACT_ARGS="--modal --gpu L4"
fi
if [ -n "$LAYERS" ]; then
    EXTRACT_ARGS="$EXTRACT_ARGS --layers $LAYERS"
fi
python scripts/02_extract_activations.py $EXTRACT_ARGS
echo ""

# Step 3: Run Analysis
echo "------------------------------------------------------------"
echo "Step 3/4: Running Analysis"
echo "------------------------------------------------------------"
python scripts/03_run_analysis.py
echo ""

# Step 4: Generate Figures
echo "------------------------------------------------------------"
echo "Step 4/4: Generating Figures"
echo "------------------------------------------------------------"
python scripts/04_generate_figures.py
echo ""

echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo ""
echo "Results:"
echo "  Dataset:    data/processed/"
echo "  Activations: data/activations/"
echo "  Analysis:   data/analysis/"
echo "  Figures:    paper/figures/"
echo ""
echo "Next steps:"
echo "  1. Review figures in paper/figures/"
echo "  2. Write paper using paper/main.tex template"
echo "  3. Submit to NewInML @ NeurIPS 2026"
