import sys
import os
from os.path import join
import argparse
import numpy as np
import pandas as pd
import anndata as ad
from torchvision.utils import make_grid
import json
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm
import subprocess
import tempfile
from mtgwas.vctest import VCTEST, VCTESTLR
from ./generator import Generator, load_image_torch, torch_imshow
import pypandoc

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
parser.add_argument('--extreme', type=float, default=0.001, help='Extreme quantile for mean embedding calculation')
parser.add_argument('--adata_path', type=str, required=True, help='Path to the anndata h5ad file with one embedding per individual in X, predicted age and score for [disease]_in_5 in obs')
parser.add_argument('--config', type=str, required=True, help='Path to the PGAN config json file')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the PGAN checkpoint file')
parser.add_argument('--outdir', type=str, required=True, help='Output directory to save results')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adata_path = args.adata_path
config = args.config
checkpoint = args.checkpoint
outdir = args.outdir
extreme = args.extreme

os.makedirs(outdir, exist_ok=True)

generator = Generator(config, checkpoint)
adata = ad.read_h5ad(adata_path)

def calculat_mean_embs(
    adata,
    id,
    outdir,
    pred_col_name=None,  
    outliers=0.0001,
    extreme=0.001
):
    if pred_col_name is None:
        pred_col_name = f"{id}_loo_pred"
    os.makedirs(outdir, exist_ok=True)
    
    values = adata.obs[pred_col_name].values
    valid_mask = ~pd.isna(values)
    valid_values = values[valid_mask]
    q1, q2, Q1, Q2 = np.quantile(valid_values, [outliers, extreme, 1 - extreme, 1 - outliers])
    print(f"Quantiles for {id}: Q1={q1}, Q2={q2}, Q1={Q1}, Q2={Q2}")

    low_score_mask = np.logical_and(values > q1, values < q2) & valid_mask
    high_score_mask = np.logical_and(values > Q1, values < Q2) & valid_mask

    embeddings_dict = {}
    for name, mask in zip(["low_score", "high_score"], [low_score_mask, high_score_mask]):
        if mask.sum() == 0:
            print(f"No cells in {name} mask for disease {id}")
            mean_emb = np.zeros(adata.X.shape[1])
        else:
            mean_emb = adata.X[mask].mean(0)
        embeddings_dict[name] = {
            "mean_embedding": mean_emb,
            "outliers": outliers,
            "extreme": extreme
        }
        np.save(join(outdir, f"{id.replace(':','_')}_{name}_mean_emb.npy"), mean_emb)
    return embeddings_dict

def plot_and_interpolate(
    generator,
    run_id,
    embeddings_dict,
    outdir,
    n_samples=20,
    n_eps=5,
    seed=args.seed,
):
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    bottom_emb = torch.tensor(
        embeddings_dict["low_score"]["mean_embedding"], dtype=torch.float32
    ).unsqueeze(0)
    top_emb = torch.tensor(
        embeddings_dict["high_score"]["mean_embedding"], dtype=torch.float32
    ).unsqueeze(0)
    left_eye = torch.tensor([[1.0, 0.0]])
    right_eye = torch.tensor([[0.0, 1.0]])

    for cond in [right_eye]:
        cond_name = "left" if cond[0, 0] == 1 else "right"
        cond_dir = os.path.join(outdir, f"{run_id}_{cond_name}_seed{seed}")
        os.makedirs(cond_dir, exist_ok=True)
        inter = torch.linspace(0, 1, n_samples)[:, None]
        eps_list = [torch.randn(1, 512) for _ in range(n_eps)]

        for i, alpha in enumerate(inter, start=1):
            emb_interp = bottom_emb * (1 - alpha) + top_emb * alpha
            cond_expand = cond.repeat(1, 1)
            x_interp = torch.cat([emb_interp, cond_expand], dim=1)
            imgs_column = []
            for eps in eps_list:
                img_tensor = generator.forward(x_interp, eps)[0]
                img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                img_uint8 = (img_np * 255).astype(np.uint8)
                imgs_column.append(img_uint8)
            col_img = np.concatenate(imgs_column, axis=1)
            Image.fromarray(col_img).save(os.path.join(cond_dir, f"{i}.png"))

all_results = []

def process_disease(id, extreme, out_dir, pred_col_name=None):
    if pred_col_name is None:
        pred_col_name = f"{id}_loo_cov_pred"
    embeddings_dict = calculat_mean_embs(
        adata,
        id,
        out_dir,
        pred_col_name=pred_col_name,
        outliers=0.0001,
        extreme=extreme
    )
    plot_and_interpolate(
        generator,
        id,
        embeddings_dict,
        out_dir,
        n_samples=20,
        n_eps=5,
        seed=args.seed
    )
    result_entry = {
        "id": id,
        "interp_images": f"{out_dir}/{id}_right_column.png",
    }
    all_results.append(result_entry)
    print(f"Finished processing disease: {id}\n\n")

exterp_dir = os.path.join(outdir, f'interpolation_in5_effect') 
os.makedirs(exterp_dir, exist_ok=True)
for disease in ['H25in_5', 'H26in_5', 'H35in_5', 'H40in_5', 'H52in_5']:
    process_disease(disease, extreme, exterp_dir, pred_col_name=f"{disease}_loo_pred")
process_disease('age', extreme, exterp_dir, pred_col_name='age_loo_pred')
