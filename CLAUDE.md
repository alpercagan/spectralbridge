# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment validation
python scripts/test_installation.py     # Validate all dependencies are installed
python scripts/test_model_loading.py    # Test BEATs/DINOv2 loading (M1 Mac)

# Install dependencies
pip install -r requirements.txt
```

No build system or test framework is configured yet. Scripts are run directly.

## Architecture

SpectralBridge aligns audio and visual signals **without text** (unlike CLIP-based approaches):

```
Audio → BEATs Encoder → 768-dim features
                               ↓
                    SpectralBridge Bridge (Fourier features + SIREN activations)
                               ↓
                    DINOv2 Feature Space → Image Generation (Phase 3)
```

**Key models**:
- **BEATs** (`BEATs_iter3_plus_AS2M.pt`): Self-supervised audio encoder. Checkpoint must be manually downloaded from [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats) and placed in `outputs/checkpoints/`.
- **DINOv2** (`vit_base_patch14_dinov2.lvd142m` via timm): Self-supervised vision encoder. Input: 518×518 images. Output: 768-dim feature vectors.
- **SpectralBridge**: Custom lightweight bridge network (~2-5M params) mapping BEATs space → DINOv2 space using Fourier feature mapping for frequency awareness.

**Target hardware**: Google Colab T4 GPU for training; Apple Silicon (MPS) for local development.

## Code Organization

- `src/models/` — Network architectures (SpectralBridge, BEATs/DINOv2 wrappers)
- `src/data/` — Dataset classes (VGGSound, Loie loaders and feature extraction)
- `src/utils/` — Caching, config loading, evaluation metrics
- `scripts/` — Standalone pipeline runners usable from Colab notebooks
- `notebooks/` — Jupyter notebooks for each development phase
- `configs/` — YAML configuration files
- `data/` — Raw/processed data and feature caches (gitignored)
- `outputs/` — Checkpoints, logs, figures (gitignored)

## Coding Style

- Type hints and Google-style docstrings on all functions and classes
- Small, focused functions with heavy caching to avoid recomputation
- Cache extracted features (BEATs/DINOv2) to disk before training

## Development Phases

1. **Week 1**: Dataset pipeline (VGGSound/Loie) + feature extraction + caching
2. **Week 2**: SpectralBridge training + retrieval evaluation
3. **Week 3**: Image generation via IP-Adapter integration
4. **Week 4**: Documentation + demo
