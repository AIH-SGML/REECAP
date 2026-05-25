import mtgwas
import pandas as pd
import numpy as np
from mtgwas.utils import gaussianize, toRanks
from mtgwas import MTGWAS
from sklearn.impute import SimpleImputer


# --- Load data ---
G = pd.read_csv("data/genotypes.bed", index_col=0).values       # genotype matrix PLINK bed file (samples × SNPs)
F = pd.read_csv("data/covariates.csv", index_col=0).values      # covariate matrix (samples × covariates)
E = pd.read_csv("data/phenotypes.csv", index_col=0).values      # phenotype matrix (samples × traits)

# --- Preprocess ---
E = gaussianize(E).astype(np.float64)
F = F.astype(np.float64)

# --- Initialize models ---
mtgwas = MTGWAS(E, F)  # Multi-trait GWAS

imputer = SimpleImputer(strategy="mean")
G = imputer.fit_transform(G).astype(np.float64)

# Run MT-GWAS on this block
mtgwas.process(G)

# Retrieve results
pvalues = mtgwas.getPv()
betas = mtgwas.getBetaSNP()
se = mtgwas.getBetaSNPste()

print("P-values:\n", pvalues)
print("Beta coefficients:\n", betas)
print("Standard errors:\n", se)
print("GWAS completed.")