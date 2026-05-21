"""
Script 2: Annotate PCA embeddings with UK Biobank phenotypes and validate
          predictive utility via leave-one-out cross-validation.

This script takes the AnnData produced by Script 1, aligns a set of UK Biobank
phenotypes (age, sex, ICD-10 disease diagnoses, polygenic risk scores) to each
participant, and assesses whether the retinal PC embeddings can predict binary
disease status and continuous age using a variance-component model (VCTESTLR from the mtgwas package)
with leave-one-out (LOO) predictions. A UMAP is also computed and saved.

Input (via command-line arguments)
-----------------------------------
--infile      : .h5ad file produced by Script 1.
--outdir      : Directory for output files.
--pc_number   : Number of PCs in the input file (used for labelling; default 40).
--seed        : Random seed (default 42).
--debug       : Drop into pdb before execution (flag).

UK Biobank data paths are hard-coded below under "File paths" and must be
adjusted to match the local UK Biobank dataset mirror. All phenotype files
are standard UK Biobank data-field CSVs (one column of values, rows indexed
by EID). The EID mapping file (eid.csv) must contain a column named 'eid'.

Required UK Biobank fields
--------------------------
Field 21003-0.0  : Age at assessment
Field 31-0.0     : Sex
Field 53-0.0     : Date of attending assessment centre
Field 130708-0.0 : Date E11 (Type 2 diabetes) first reported
Field 131182-0.0 : Date H35 (Other retinal disorders) first reported
Field 131186-0.0 : Date H40 (Glaucoma) first reported
Field 131164-0.0 : Date H25 (Age-related cataract) first reported
Field 131166-0.0 : Date H26 (Other cataract) first reported
Field 20262-0.0  : Myopia diagnosis
Field 131208-0.0 : Date H52 (Disorders of refraction) first reported
Field 131212-0.0 : Date H54 (Blindness and low vision) first reported
Field 131178-0.0 : Date H33 (Retinal detachments) first reported
Field 131184-0.0 : Date H36 (Retinal disorders in diseases elsewhere) first reported
PRS file         : prs_scores.csv — fields 26265-0.0 (glaucoma PRS),
                   26204-0.0 (AMD PRS)

Output
------
<outdir>/<pc_number>_PC_tuned_averaged2eyes_rpupd_filled_obs.h5ad
    Annotated AnnData with phenotype columns, binary disease flags,
    LOO predictions, UMAP coordinates, and PRS scores.

<outdir>/auc_loo_pheno_preds.png
    Bar chart of AUC (binary traits) for LOO predictions.

umap_representation_<pc_number>PCs.png
    UMAP scatter plot coloured by age.

Usage example
-------------
python 02_annotate_and_validate.py \\
    --infile  /path/to/40_PC_tuned_averaged2eyes_rand_path.h5ad \\
    --outdir  /path/to/output/ \\
    --pc_number 40 \\
    --seed 42
"""

import argparse
import os
import random
from os.path import join

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mtgwas.vctest import VCTESTLR
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.utils import check_random_state
from umap import UMAP


# ---------------------------------------------------------------------------
# Helpers — EID extraction
# ---------------------------------------------------------------------------

def extract_eid(file_path: str) -> str:
    """Return the participant EID encoded in the image filename.

    Expected filename format: <eid>_<field>_<...>.jpg  (or any extension).
    """
    return os.path.basename(file_path).split("_")[0]


# ---------------------------------------------------------------------------
# Helpers — phenotype alignment
# ---------------------------------------------------------------------------

def align_phenotype_to_eids(adata, phenofile: str, column_name: str, eid_mapping: pd.DataFrame):
    """Load a UK Biobank phenotype CSV and align it to the AnnData obs index.

    Parameters
    ----------
    adata        : AnnData object whose obs contains an 'eid' column.
    phenofile    : Path to a UKB phenotype CSV (rows = participants, no header
                   index — EIDs are supplied via eid_mapping).
    column_name  : Name of the new column to add to adata.obs.
    eid_mapping  : DataFrame with an 'eid' column used as the row index for
                   the phenotype file.
    """
    pheno = pd.read_csv(phenofile).set_index(eid_mapping["eid"])
    pheno.index = pheno.index.astype(str)
    aligned = pheno.reindex(adata.obs["eid"])
    adata.obs[column_name] = aligned.values.flatten()


# ---------------------------------------------------------------------------
# Helpers — VCTESTLR pipeline
# ---------------------------------------------------------------------------

def preprocess_data(adata, target_key: str):
    """Extract feature matrix X and target vector Y, dropping rows with NaN in Y.

    Parameters
    ----------
    adata      : AnnData object.
    target_key : Column in adata.obs used as the prediction target.

    Returns
    -------
    X    : Feature matrix (n_valid x n_features), dense float array.
    Y    : Target vector (n_valid x 1).
    mask : Boolean array of length n_obs; True where Y is not NaN.
    """
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    Y = adata.obs[target_key].values.reshape(-1, 1)
    mask = ~np.isnan(Y).flatten()
    return X[mask], Y[mask], mask


def fit_vctestlr(X: np.ndarray, Y: np.ndarray, ndeltas: int = 100) -> VCTESTLR:
    """Fit a VCTESTLR variance-component model.

    Parameters
    ----------
    X       : Feature matrix.
    Y       : Target vector.
    ndeltas : Grid size for the delta (signal-to-noise) parameter search.

    Returns
    -------
    Fitted VCTESTLR model.
    """
    model = VCTESTLR(ndeltas=ndeltas)
    model.fit(X, Y, normalize_X=False, compute_pvals=False, verbose=True)
    return model


def add_loo_predictions(adata, model: VCTESTLR, mask: np.ndarray,
                        target_key: str, pred_key: str):
    """Compute LOO predictions and store them in adata.obs.

    The VCTESTLR model returns mean-centred LOO residuals; the phenotype mean
    (computed over all observations, including those excluded due to missing
    data) is added back so that predictions are on the original scale.

    Parameters
    ----------
    adata      : AnnData object.
    model      : Fitted VCTESTLR model.
    mask       : Boolean mask identifying rows used during model fitting.
    target_key : Column in adata.obs that was used as the target.
    pred_key   : Column name under which predictions will be stored.
    """
    loo_preds = model.predict_loo().flatten()
    target_mean = adata.obs[target_key].mean()
    adata.obs.loc[mask, pred_key] = loo_preds + target_mean


def predict_ystar_vclr(adata, target_key: str, pred_key: str, ndeltas: int = 100):
    """End-to-end pipeline: preprocess → fit VCTESTLR → store LOO predictions.

    Parameters
    ----------
    adata      : AnnData object.
    target_key : Column in adata.obs used as the prediction target.
    pred_key   : Column name under which LOO predictions will be stored.
    ndeltas    : Grid size for the VCTESTLR delta search (default: 100).
    """
    X, Y, mask = preprocess_data(adata, target_key)
    model = fit_vctestlr(X, Y, ndeltas=ndeltas)
    add_loo_predictions(adata, model, mask, target_key, pred_key)


# ---------------------------------------------------------------------------
# Helpers — evaluation and plotting
# ---------------------------------------------------------------------------

def calculate_auc_and_plot(df: pd.DataFrame, target_cols: list, outplot: str,
                           pred_col_suffix: str = "_loo_pred",
                           problem_type: str = "binary"):
    """Compute performance metrics for LOO predictions and save a bar chart.

    Parameters
    ----------
    df             : DataFrame containing both target and prediction columns.
    target_cols    : List of target column names.
    outplot        : File path for the output figure.
    pred_col_suffix: Suffix appended to each target column name to obtain the
                     corresponding prediction column (default: '_loo_pred').
    problem_type   : One of 'binary' (AUC-ROC), 'regression' (R²), or
                     'multiclass' (macro-averaged AUC-ROC).
    """
    results = []

    for target_key in target_cols:
        pred_key = f"{target_key}{pred_col_suffix}"

        if target_key not in df.columns or pred_key not in df.columns:
            print(f"Warning: column '{target_key}' or '{pred_key}' not found — skipping.")
            continue

        valid = df[[target_key, pred_key]].dropna()
        if valid.empty:
            print(f"Warning: no valid rows for '{target_key}' — skipping.")
            continue

        if problem_type == "binary":
            score = roc_auc_score(valid[target_key], valid[pred_key])
        elif problem_type == "regression":
            score = r2_score(valid[target_key], valid[pred_key])
        elif problem_type == "multiclass":
            score = roc_auc_score(
                valid[target_key], valid[pred_key],
                average="macro", multi_class="ovr",
            )
        else:
            raise ValueError(
                f"Unsupported problem_type '{problem_type}'. "
                "Choose from: 'binary', 'regression', 'multiclass'."
            )

        results.append((target_key, score))

    if not results:
        print("No results to plot.")
        return

    auc_df = pd.DataFrame(results, columns=["Target", "Score"])

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(auc_df["Target"], auc_df["Score"], color="skyblue")
    ax.set_xlabel("Target variable")
    ax.set_ylabel("AUC (binary / multiclass) or R² (regression)")
    ax.set_title("LOO prediction performance")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(outplot, dpi=300)
    plt.close()
    print(f"Performance bar chart saved to: {outplot}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Annotate retinal PC embeddings with UK Biobank phenotypes and "
            "validate predictive utility via LOO cross-validation."
        )
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--pc_number", type=int, default=40,
                        help="Number of PCs in the input AnnData (default: 40).")
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to .h5ad file produced by Script 1.")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--debug", dest="debug", action="store_true", default=False,
                        help="Drop into pdb at the start of main().")
    args = parser.parse_args()

    if args.debug:
        import pdb
        pdb.set_trace()

    # Reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    check_random_state(seed)

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Paths — adjust to match your local UK Biobank mirror
    # ------------------------------------------------------------------
    phenodir = "/path/to/ukbb/dataset/"          # <-- set this
    project_dir = "/path/to/project/"            # <-- set this

    eid_file             = join(phenodir, "phenos/eid.csv")
    age_file             = join(phenodir, "phenos/21003-0.0.csv")
    sex_file             = join(phenodir, "phenos/31-0.0.csv")
    assessment_date_file = join(phenodir, "phenos/53-0.0.csv")

    # First-occurrence fields (UKB algorithmically-defined outcomes)
    E11_first_diagnosed = join(phenodir, "phenos/130708-0.0.csv")   # Type 2 diabetes
    H35_first_diagnosed = join(phenodir, "phenos/131182-0.0.csv")   # Other retinal disorders
    H40_first_diagnosed = join(phenodir, "phenos/131186-0.0.csv")   # Glaucoma
    H25_first_diagnosed = join(phenodir, "phenos/131164-0.0.csv")   # Age-related cataract
    H26_first_diagnosed = join(phenodir, "phenos/131166-0.0.csv")   # Other cataract
    myopia_diag         = join(phenodir, "phenos/20262-0.0.csv")    # Myopia diagnosis code
    H52_first_diagnosed = join(phenodir, "phenos/131208-0.0.csv")   # Disorders of refraction
    H54_first_diagnosed = join(phenodir, "phenos/131212-0.0.csv")   # Blindness / low vision
    H33_first_diagnosed = join(phenodir, "phenos/131178-0.0.csv")   # Retinal detachment
    H36_first_diagnosed = join(phenodir, "phenos/131184-0.0.csv")   # Diabetic retinopathy

    prs_file = join(phenodir, "prs_scores.csv")  # Polygenic risk scores

    # ------------------------------------------------------------------
    # Load AnnData and extract EIDs from image paths
    # ------------------------------------------------------------------
    pc_adata = ad.read_h5ad(args.infile)
    pc_adata.obs["eid"] = pc_adata.obs["path"].apply(extract_eid)

    updated_adata_file = join(
        args.outdir,
        f"{args.pc_number}_PC_tuned_averaged2eyes_rpupd_filled_obs.h5ad",
    )

    # ------------------------------------------------------------------
    # Align phenotypes
    # ------------------------------------------------------------------
    eid_mapping = pd.read_csv(eid_file)

    phenotype_files = {
        "age":                  age_file,
        "sex":                  sex_file,
        "assessment_date":      assessment_date_file,
        "E11_first_diagnosed":  E11_first_diagnosed,
        "H35_first_diagnosed":  H35_first_diagnosed,
        "H40_first_diagnosed":  H40_first_diagnosed,
        "H25_first_diagnosed":  H25_first_diagnosed,
        "H26_first_diagnosed":  H26_first_diagnosed,
        "myopia_diag":          myopia_diag,
        "H52_first_diagnosed":  H52_first_diagnosed,
        "H54_first_diagnosed":  H54_first_diagnosed,
        "H33_first_diagnosed":  H33_first_diagnosed,
        "H36_first_diagnosed":  H36_first_diagnosed,
    }

    for column_name, file_path in phenotype_files.items():
        align_phenotype_to_eids(pc_adata, file_path, column_name, eid_mapping)

    # ------------------------------------------------------------------
    # Create binary disease flags and 5-year incidence columns
    # ------------------------------------------------------------------
    disease_date_columns = [
        "E11_first_diagnosed",
        "H35_first_diagnosed",
        "H40_first_diagnosed",
        "H25_first_diagnosed",
        "H26_first_diagnosed",
        "H52_first_diagnosed",
        "H54_first_diagnosed",
        "H33_first_diagnosed",
        "H36_first_diagnosed",
    ]

    pc_adata.obs["assessment_date"] = pd.to_datetime(
        pc_adata.obs["assessment_date"], errors="coerce"
    )

    bin_cols = []   # ever-diagnosed binary flags
    in5_cols = []   # incident-within-5-years binary flags
    for col in disease_date_columns:
        if col.endswith("_date"):
            # ICD-10 HES-derived date: any record = prevalent case
            bin_col = col.replace("_date", "_bin")
        elif col.endswith("_first_diagnosed"):
            # UKB first-occurrence field: classify relative to assessment date
            pc_adata.obs[col] = pd.to_datetime(pc_adata.obs[col], errors="coerce")
            in5_col = col.replace("_first_diagnosed", "_in5")

            # Three-way classification:
            #   None  — diagnosis predates assessment (prevalent case, excluded)
            #   1     — diagnosed within 5 years after assessment (incident case)
            #   0     — not diagnosed within 5 years (control)
            pc_adata.obs[in5_col] = pc_adata.obs.apply(
                lambda row: (
                    None
                    if pd.notna(row[col]) and row[col] < row["assessment_date"]
                    else (
                        1
                        if pd.notna(row[col])
                        and row["assessment_date"]
                        <= row[col]
                        <= row["assessment_date"] + pd.DateOffset(years=5)
                        else 0
                    )
                ),
                axis=1,
            )
            in5_cols.append(in5_col)

            # LOO prediction for incident disease within 5 years
            predict_ystar_vclr(pc_adata, target_key=in5_col, pred_key=f"{in5_col}_loo_pred")

            bin_col = col.replace("_first_diagnosed", "_bin")
        else:
            bin_col = col + "_bin"

        # Binary flag: 1 if any date is recorded (ever diagnosed), else 0
        pc_adata.obs[bin_col] = pc_adata.obs[col].notna().astype(int)
        bin_cols.append(bin_col)

        # LOO prediction for ever-diagnosed status
        predict_ystar_vclr(pc_adata, target_key=bin_col, pred_key=f"{bin_col}_loo_pred")

    # Convert remaining datetime columns to strings for HDF5 compatibility
    for col in pc_adata.obs.select_dtypes(include=["datetime64[ns]", "datetime"]).columns:
        pc_adata.obs[col] = pc_adata.obs[col].astype(str)

    # ------------------------------------------------------------------
    # AUC bar charts for LOO predictions
    # ------------------------------------------------------------------
    # Ever-diagnosed (prevalent) binary flags
    outplot_bin = join(args.outdir, "auc_loo_pheno_preds_ever_diagnosed.png")
    calculate_auc_and_plot(
        pc_adata.obs, bin_cols, outplot_bin,
        pred_col_suffix="_loo_pred", problem_type="binary",
    )

    # Incident disease within 5 years of assessment
    if in5_cols:
        outplot_in5 = join(args.outdir, "auc_loo_pheno_preds_in5years.png")
        calculate_auc_and_plot(
            pc_adata.obs, in5_cols, outplot_in5,
            pred_col_suffix="_loo_pred", problem_type="binary",
        )

    # ------------------------------------------------------------------
    # LOO prediction for age (continuous)
    # ------------------------------------------------------------------
    predict_ystar_vclr(pc_adata, target_key="age", pred_key="age_loo_pred")

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------
    pc_adata.X = pc_adata.X.astype(np.float32)
    umap_model = UMAP(n_components=2, random_state=seed)
    pc_adata.obsm["X_umap"] = umap_model.fit_transform(pc_adata.X)
    umap_embedding = pc_adata.obsm["X_umap"]

    # Save annotated AnnData before plotting (so UMAP is captured)
    pc_adata.write_h5ad(updated_adata_file)
    print(f"Annotated AnnData saved to: {updated_adata_file}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        umap_embedding[:, 0], umap_embedding[:, 1],
        c=pc_adata.obs["age"].values.astype(float),
        cmap="bone", s=5,
    )
    plt.colorbar(sc, ax=ax, label="Age")
    ax.set_title(f"UMAP of {args.pc_number} PCs coloured by age")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    umap_outpath = join(args.outdir, f"umap_representation_{args.pc_number}PCs.png")
    plt.tight_layout()
    plt.savefig(umap_outpath, dpi=300)
    plt.close()
    print(f"UMAP plot saved to: {umap_outpath}")

    # ------------------------------------------------------------------
    # Add polygenic risk scores (PRS)
    # ------------------------------------------------------------------
    prs = pd.read_csv(prs_file, usecols=["eid", "26265-0.0", "26204-0.0"])
    prs = prs.rename(columns={"26265-0.0": "glaucoma_prs", "26204-0.0": "amd_prs"})
    prs["eid"] = prs["eid"].astype(str)
    prs = prs.set_index("eid")

    pc_adata.obs["eid"] = pc_adata.obs["eid"].astype(str)
    pc_adata.obs = pc_adata.obs.merge(prs, how="left", left_on="eid", right_index=True)

    pc_adata.write_h5ad(updated_adata_file)
    print(f"Final annotated AnnData (with PRS) saved to: {updated_adata_file}")


if __name__ == "__main__":
    main()
