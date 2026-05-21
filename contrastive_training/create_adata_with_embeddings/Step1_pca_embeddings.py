"""
Script 1: Average retinal embeddings across eyes and compute PCA.

This script takes fine-tuned RETFound embeddings (1024-dimensional) for retinal
fundus images, averages the representations across both eyes per participant,
and reduces dimensionality via PCA. The result is saved as an AnnData (.h5ad) file.

Input
-----
--merged_df_file : CSV file containing per-image embeddings and their file paths.
    Expected columns:
        - file_path : path to the retinal image, with filename format <eid>_<field>_...
        - embedding_0 ... embedding_1023 : RETFound embedding dimensions

--out_folder     : Directory where outputs are written.
--number_of_PCs  : Number of principal components to retain (default: 40).
--seed           : Random seed for reproducibility (default: 42).
--debug          : Drop into pdb before execution (flag).

Output
------
<out_folder>/<number_of_PCs>_PC_tuned_averaged2eyes_rand_path.h5ad
    AnnData object where:
        - .X      : PCA-transformed embeddings (n_participants x n_PCs)
        - .obs    : participant metadata; 'path' column stores the original image path
        - .var    : PC labels (PC1 ... PCn)

<out_folder>/<number_of_PCs>_PC_explained_variance.txt
    Per-component and cumulative explained variance.

Replication (UK Biobank)
------------------------
Fine-tune RETFound (Zhou et al., 2023; https://github.com/rmaphoh/RETFound_MAE)
on UK Biobank fundus images using age as the supervision signal. Extract
1024-dimensional embeddings with average pooling for each image and save them together
with the image file paths in a single CSV (one row per image). Pass that CSV
as --merged_df_file.

Usage example
-------------
python 01_pca_embeddings.py \\
    --merged_df_file /path/to/embeddings.csv \\
    --out_folder    /path/to/output/ \\
    --number_of_PCs 40 \\
    --seed          42
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import anndata
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_eid(file_path: str) -> str:
    """Return the participant EID encoded in the image filename.

    Expected filename format: <eid>_<field>_<...>.jpg  (or any extension).
    """
    filename = os.path.basename(file_path)
    return filename.split("_")[0]


def extract_field(file_path: str) -> str:
    """Return the UK Biobank field ID encoded in the image filename.

    Expected filename format: <eid>_<field>_<...>.jpg  (or any extension).
    """
    filename = os.path.basename(file_path)
    return filename.split("_")[1]


def get_random_path(group):
    """Sample one file path from a participant's group (used to retain a
    representative path index after averaging across eyes)."""
    return group.sample(n=1).index[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    check_random_state(args.seed)

    os.makedirs(args.out_folder, exist_ok=True)

    if args.debug:
        breakpoint()

    # ------------------------------------------------------------------
    # Load embeddings
    # ------------------------------------------------------------------
    merged_df = pd.read_csv(args.merged_df_file)
    merged_df["eid"] = merged_df["file_path"].apply(extract_eid)
    merged_df["field"] = merged_df["file_path"].apply(extract_field)
    merged_df = merged_df.set_index("file_path")

    embedding_cols = [c for c in merged_df.columns if c.startswith("embedding")]
    if not embedding_cols:
        raise ValueError(
            "No columns starting with 'embedding' found in --merged_df_file. "
            "Expected columns named embedding_0, embedding_1, ..."
        )

    # ------------------------------------------------------------------
    # Average embeddings across eyes (left / right) per participant
    # ------------------------------------------------------------------
    avg_embeddings = (
        merged_df[embedding_cols + ["eid"]]
        .groupby("eid")[embedding_cols]
        .mean()
    )

    # Retain one representative image path per participant as the obs index
    random_paths = merged_df.groupby("eid").apply(get_random_path)
    avg_embeddings.index = random_paths.values

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    pca = PCA(
        n_components=args.number_of_PCs,
        svd_solver="full",
        random_state=args.seed,
    )
    pca_result = pca.fit_transform(avg_embeddings)

    # ------------------------------------------------------------------
    # Save as AnnData
    # ------------------------------------------------------------------
    pc_labels = [f"PC{i + 1}" for i in range(args.number_of_PCs)]
    adata = anndata.AnnData(
        X=pca_result.astype(np.float32),
        obs=pd.DataFrame(index=avg_embeddings.index),
        var=pd.DataFrame(index=pc_labels),
    )
    adata.obs["path"] = adata.obs.index  # store original image path as metadata

    out_h5ad = os.path.join(
        args.out_folder,
        f"{args.number_of_PCs}_PC_tuned_averaged2eyes_rand_path.h5ad",
    )
    adata.write(out_h5ad)
    print(f"AnnData saved to: {out_h5ad}")

    # ------------------------------------------------------------------
    # Save explained variance
    # ------------------------------------------------------------------
    explained_var = pca.explained_variance_ratio_ * 100
    lines = [f"PC{i + 1}: {v:.4f}%" for i, v in enumerate(explained_var)]
    lines.append(f"\nTotal variance explained ({args.number_of_PCs} PCs): {explained_var.sum():.4f}%")
    variance_info = "\n".join(lines)

    print(variance_info)

    out_var = os.path.join(
        args.out_folder,
        f"{args.number_of_PCs}_PC_explained_variance.txt",
    )
    with open(out_var, "w") as f:
        f.write(variance_info)
    print(f"Explained variance saved to: {out_var}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Average RETFound retinal embeddings across both eyes per participant "
            "and reduce to principal components."
        )
    )
    parser.add_argument(
        "--merged_df_file",
        type=str,
        required=True,
        help="Path to CSV with per-image embeddings and file paths.",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        required=True,
        help="Output directory for AnnData and variance files.",
    )
    parser.add_argument(
        "--number_of_PCs",
        type=int,
        default=40,
        help="Number of principal components to retain (default: 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Drop into pdb debugger at the start of main().",
    )
    main(parser.parse_args())
