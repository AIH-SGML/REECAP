# Generative Model for REECAP

This repository provides **generative model** used in the **REECAP** project for visualizing medical image embeddings via **Progressive GANs (PGAN)**.  
It enables **reproducible training, evaluation, and visualization** of tissue-specific generative models.

PGAN is the main implemented model, with the implementation adapted from [Facebook’s pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo).

---

## Repository Structure
```
generative_model/
│
├── PGAN/               # Main PGAN training pipeline and configs
│   ├── models/         # Model architectures, losses, and utilities
│   ├── visualization/  # Tools for sample plots
│   ├── experiments/    # training and latent space visualization scripts
│   └── config/         # configs for PGAN training
│
└── data/             # Example of the data input for model training and latent space visualization
```

---

## User Guide

- **./PGAN/README.md** — model training instructions  
- **./PGAN/models/README.md** — architecture details

---

## Quick Start

### 1. Installation

```
conda env create -f genmodel.yml 
conda activate PGAN
```

### 2. Prepare data
Data is expected in AnnData format. We provide a small example dataset.

```
data/
  example_embeddings.h5ad
```
---

## Main Scripts

### Train the generative model

```
python PGAN/experiments/train_PGAN.py
```

### Disease reconstruction experiment

```
python PGAN/experiments/reconstruct_disease_in_5_effect.py
```
---

## Configuration

Main configuration file:

```
PGAN/config/config.json
```
---

## Outputs

Outputs are saved to:

```
output/
```

---

## Attribution

Based on [Facebook’s pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo). Extended in Chaudhary et al, 2024. Here adapted for REECAP.

