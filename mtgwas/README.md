# MT-GWAS

**MT-GWAS** (Multi-Trait Genome-Wide Association Study) is a Python package for performing genome-wide association analyses across multiple traits.  
It provides efficient implementations for multi-trait linear mixed models, variance component (VC) tests, and related statistical tools.

---

## Package Overview

The `mtgwas` package includes modules for running association tests, computing statistical metrics, and performing genome-wide analyses on multi-trait datasets.

### Modules

| Module | Description |
|---------|--------------|
| `mtgwas.py` | Main entry point for performing multi-trait GWAS analysis. Handles model fitting, inference, and output of association statistics. |
| `vctest.py` | Implements variance component tests used to evaluate the significance of genetic effects. |
| `gwas.py` | Core utilities for GWAS computations, including data handling, test statistics, and likelihood calculations. |
| `stats.py` | Statistical helper functions for model fitting, p-value computation, and other related statistical tests. |
| `__init__.py` | Initializes the package and exposes primary interfaces. |

### Input Format
- **Genotypes**: CSV or PLINK-style file with individuals in rows and SNPs in columns.
- **Traits**: CSV file with matching sample IDs and numeric trait columns.
---

## Installation

This package is expected to be installed and run on a Linux environment.

1. Install the required dependencies before running the package:

```
conda create -n mtgwas python=3.11
conda activate mtgwas

conda install -c conda-forge \
numpy=1.25.2 \
pandas=2.1.0 \
scipy=1.11.2 \
matplotlib=3.7.2 \
scikit-learn=1.3.0 \
statsmodels=0.14.0 \
pytorch=2.2.0 \
tqdm=4.66.1 \
limix-core=1.0.2 \
chiscore=0.2.2 

pip install limix-lmm==0.1.2 statsmodels==0.14.0 torch
```

2. Install the package locally in editable mode:

```
pip install -e .
```
---

## Example: Running MT-GWAS in Python

The example below demonstrates how to run a multi-trait GWAS using MT-GWAS when genotype, covariate, and phenotype data are provided as CSV files. Computations take ~2 min with 12 CPU available.

```
python example_run.py
```

