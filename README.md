# The Geometry of Untranslatability

**Probing LLM Representations of Cultural Concepts Without English Equivalents**

Target venue: [NewInML @ NeurIPS 2026](https://newinml.github.io/NewInML2026NeurIPS/) (Paris, France)

---

## Overview

Recent work has shown that multilingual LLMs route through an English-centric latent space for semantic processing. But what happens when a cultural concept **has no English equivalent**?

This project investigates how LLMs internally represent untranslatable cultural concepts like:
- Japanese *Tsundoku* (積ん読) — buying books and letting them pile up unread
- German *Schadenfreude* — pleasure derived from another's misfortune
- Portuguese *Saudade* — deep emotional longing for something absent
- Korean *Han* (한) — collective grief from historical oppression

We use mechanistic interpretability tools (residual stream analysis, logit lens, activation clustering) on **Llama-3-8B** to analyze 70 untranslatable + 82 translatable control concepts across 10 languages.

## Key Research Questions

1. **English Routing:** Does the model force untranslatable concepts into its English-centric latent space?
2. **Clustering:** Do untranslatable concepts occupy isolated, language-specific clusters vs. overlapping with English-centric representations?
3. **Trajectory:** How do activation trajectories differ between translatable and untranslatable concepts across layers?
4. **Generation Quality:** Does forced English alignment correlate with worse cultural explanations?

The dataset contains **70 untranslatable concepts** and **82 translatable control concepts** across 10 languages, totaling **152 concepts** and **456 prompts** (3 templates per concept).

## Project Structure

```
Culture-Geo/
├── configs/
│   └── default.yaml              # Experiment configuration
├── data/
│   ├── raw/                      # Raw data (gitignored)
│   ├── processed/                # Built dataset (concepts.json, prompts.json)
│   └── activations/              # Extracted activations (.npz files)
├── src/
│   ├── dataset/
│   │   ├── concepts.py           # 200 untranslatable + 200 translatable concepts
│   │   ├── prompts.py            # 3 prompt templates per concept
│   │   └── build_dataset.py      # Dataset construction script
│   ├── extraction/
│   │   ├── hooks.py              # HuggingFace activation hooks
│   │   ├── extract.py            # Local GPU extraction pipeline
│   │   └── modal_extract.py      # Modal serverless extraction
│   ├── analysis/
│   │   ├── logit_lens.py         # English Routing Rate computation
│   │   ├── clustering.py         # PCA, UMAP, k-means evaluation
│   │   ├── trajectory.py         # Layer-wise trajectory tracking
│   │   └── statistics.py         # Welch's t-tests, effect sizes
│   ├── visualization/
│   │   ├── plots.py              # All matplotlib figure functions
│   │   └── figures.py            # Paper figure generation
│   └── utils/
│       ├── io.py                 # Data loading/saving utilities
│       └── metrics.py            # Translation quality metrics
├── scripts/
│   ├── 01_build_dataset.py       # Step 1: Build dataset
│   ├── 02_extract_activations.py # Step 2: Extract activations
│   ├── 03_run_analysis.py        # Step 3: Run all analyses
│   ├── 04_generate_figures.py    # Step 4: Generate paper figures
│   └── 05_run_all.sh            # Run complete pipeline
├── paper/
│   └── main.tex                  # LaTeX paper template (NeurIPS format)
├── tests/
│   ├── test_dataset.py
│   └── test_extraction.py
├── requirements.txt              # Python dependencies
├── requirements-modal.txt        # Modal-specific dependencies
├── pyproject.toml                # Project configuration
├── .gitignore
└── README.md                     # This file
```

## Quick Start

### Prerequisites

- Python 3.10+
- GPU with 16GB+ VRAM (for local) OR Modal account (for serverless)
- HuggingFace access to [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/Culture-Geo.git
cd Culture-Geo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Option A: Local GPU
bash scripts/05_run_all.sh

# Option B: Modal serverless (recommended, ~$10-15 total)
bash scripts/05_run_all.sh --modal

# Option C: Step by step
python scripts/01_build_dataset.py
python scripts/02_extract_activations.py --model meta-llama/Meta-Llama-3-8B
python scripts/03_run_analysis.py
python scripts/04_generate_figures.py
```

## Step-by-Step Guide

### Step 1: Build Dataset

```bash
python scripts/01_build_dataset.py
# or with options
python scripts/01_build_dataset.py --num-concepts-per-language 15 --seed 42
```

Output:
- `data/processed/concepts.json` — 200 untranslatable + 200 translatable concepts
- `data/processed/prompts.json` — 600 formatted prompts
- `data/processed/manifest.json` — Dataset statistics

### Step 2: Extract Activations

**Local GPU:**
```bash
python scripts/02_extract_activations.py \
    --model meta-llama/Meta-Llama-3-8B \
    --device cuda \
    --batch-size 8 \
    --max-length 128
```

**Modal Serverless (L4 GPU at $0.80/hr):**
```bash
# First, set up Modal
pip install modal
modal setup

# Run extraction
python scripts/02_extract_activations.py --modal --gpu L4
```

**Memory-efficient (specific layers only):**
```bash
python scripts/02_extract_activations.py --layers 8,12,16,20,24,28,32
```

Output:
- `data/activations/layer_XX.npz` — Activations per layer
- `data/activations/logits.npz` — Final logits
- `data/activations/metadata.json` — Prompt metadata

### Step 3: Run Analysis

```bash
python scripts/03_run_analysis.py
# or with options
python scripts/03_run_analysis.py \
    --activations-dir data/activations \
    --output-dir data/analysis \
    --pca-components 50 \
    --n-clusters 10
```

Output:
- `data/analysis/analysis_results.json` — All analysis results

### Step 4: Generate Figures

```bash
python scripts/04_generate_figures.py
```

Output (6 paper figures):
- `paper/figures/fig1_umap_translatability.pdf` — UMAP embedding by translatability
- `paper/figures/fig1b_umap_language.pdf` — UMAP embedding by language
- `paper/figures/fig2_err_across_layers.pdf` — English Routing Rate
- `paper/figures/fig3_trajectory_distances.pdf` — Trajectory divergence
- `paper/figures/fig4_ari_by_layer.pdf` — Clustering quality (ARI)
- `paper/figures/fig5_silhouette_by_group.pdf` — Cluster separation
- `paper/figures/fig6_statistical_results.pdf` — Effect sizes

## Compute Budget

| Component | GPU | Time | Cost |
|-----------|-----|------|------|
| Activation extraction (600 prompts) | Modal L4 | ~30 min | ~$0.40 |
| Extra experiments (logit lens, etc.) | Modal L4 | ~2-3 hrs | ~$2.00 |
| Debugging / re-runs | Modal L4 | ~2-3 hrs | ~$2.00 |
| Buffer | Modal L4 | ~2 hrs | ~$1.60 |
| **Total** | | **~8 hrs** | **~$6.00** |

Using Modal's free $30/month credit, this project costs effectively **$0**.

## Dataset Details

### Untranslatable Concepts (70)

| Language | Count | Examples |
|----------|-------|----------|
| Japanese | 10 | Tsundoku, Wabi-sabi, Mono no aware, Komorebi, Kintsugi |
| German | 10 | Schadenfreude, Weltschmerz, Fernweh, Waldeinsamkeit, Zeitgeist |
| Portuguese | 8 | Saudade, Desenrascanço, Sobremesa, Cafuné, Madrugada |
| Korean | 8 | Jeong, Han, Nunchi, Palli-palli, Chemyon |
| Arabic | 6 | Tarab, Ya'aburnee, Taza, Khayyal, Samar |
| Hindi | 6 | Jugaad, Garma-garam, Dosti, Sukoon, Dil |
| Russian | 5 | Toska, Pochemuchka, Davai, Zapoi |
| Danish | 6 | Hygge, Forelsket, Pyt, Lykke, Trivsel |
| Spanish | 6 | Sobremesa, Estrenar, Duende, Convivencia |
| French | 5 | Depaysement, L'esprit de l'escalier, Flâneur, Retrouvailles |

### Translatable Controls (82)

Matched by language and frequency. Examples: Japanese *Ryokan* (= inn), German *Kindergarten* (= preschool), Portuguese *Samba* (= dance).

### Prompt Templates (3 per concept)

1. **Definition:** "In {language}, the word {word} means..."
2. **Usage:** "Write a sentence in {language} using the word {word}..."
3. **Comparison:** "Explain the {language} concept {word} to someone who only speaks English..."

## Analysis Methods

### Logit Lens / English Routing Rate

Project intermediate representations through the unembedding matrix to obtain layer-wise token predictions. Compute the fraction of top-1 English tokens at each layer.

### PCA + UMAP Clustering

Reduce 4096-dimensional activations to 50 principal components, then to 2D via UMAP for visualization. Evaluate clustering quality with k-means (k=10) using Adjusted Rand Index and Silhouette Score.

### Trajectory Analysis

Track cosine distance from the translatable-concept centroid at each layer. Higher distance = more divergence from English-adjacent representation space.

### Statistical Testing

Welch's t-test with Cohen's d effect sizes and 95% confidence intervals for all comparisons between untranslatable and translatable groups.

## Related Work

- **CuE** (Khanuja et al., 2026) — Cultural Embeddings via SAEs for steering
- **"Do Multilingual LLMs Think In English?"** (Schut et al., 2025) — English-centric latent spaces
- **"Deciphering Cultural Representations"** (Zou et al., ACL 2026) — SAE-based cultural feature analysis
- **RomanLens** (2025) — Latent romanization as bridge between concept and language-specific space

## Paper

The paper is being prepared for submission to **NewInML @ NeurIPS 2026**:
- **Deadline:** August 29, 2026
- **Format:** 2-8 pages (excluding references), NeurIPS workshop template
- **Template:** See `paper/main.tex`

## License

MIT License. See `LICENSE` for details.

## Citation

```bibtex
@inproceedings{culturegeo2026,
    title={The Geometry of Untranslatability: How LLMs Represent Cultural Concepts Without English Equivalents},
    author={Anonymous},
    booktitle={Workshop on New In Machine Learning (NeurIPS 2026)},
    year={2026}
}
```

## Acknowledgments

Built for the NewInML @ NeurIPS 2026 workshop. Uses Llama-3-8B by Meta.
