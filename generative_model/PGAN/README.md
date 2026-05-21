# PGAN Decoder for REECAP

This module is adapted version of HistoGWAS, a framework which fine-tunes a Progressive GAN (PGAN) decoder on embeddings derived from medical images. The decoder learns to reconstruct the fundus given the the embedding, eye laterality (left or right) and randomly sampled noise. The trained generator is used to visualise clusters, interpolate between embeddings, and support downstream gene-prediction analyses.

---

## Repository Structure
- `config/` – sample JSON configs providing dataset locations and training hyperparameters (e.g. `config.json`).
- `experiments/train_PGAN.py` – code launcher that creates per-tissue configs and starts PGAN jobs.
- `datasets.py` – AnnData-backed dataset loader for fundus embeddings.
- `models/`, `visualization/` – forked components from the Facebook `pytorch_GAN_zoo` implementation.
- `experiments/` – default location for logs and checkpoints produced by manual runs (`train_PGAN.py`) and the generated images for the extreme cases of a predicted disease in 5 years score (`reconstruct_disease_in_5_effect.py`).
- `generator.py` - a class to generate single image given the traned model and embedding.
- `../output/` – default output root.

---

## Installation
- Create and activate the conda environment defined in `genmodel.yml` from the `generative_model/` directory (e.g. `conda env create -f genmodel.yml` and `conda activate PGAN`).
- Python environment satisfying `requirements.txt` plus the broader HistoGWAS dependencies (PyTorch ≥1.10, torchvision, h5py, AnnData/scanpy stack).
- Access to AnnData `.h5ad` files containing tile embeddings and metadata. Each file must expose:
  - `adata.X`: numeric embedding matrix (samples × embedding dimension).
  - `adata.obs`: metadata, including `path` with image paths for visualisation and columns `eye_left` and `eye_right` identifying laterality of the eye.
- Editable installations of `mtgwas` from the repository root (`pip install -e ...`).

---

## Launch Training (Local GPU)
Use the shell wrapper when running on a workstation or interactive GPU node (adapt the paths in train_PGAN.py to your dataset):
```
cd ./PGAN/experiments
python train_PGAN.py
```
Expected output: 

`reecap_s[1-5]_i[16000].pt` – model weights for each stage and resolution  

Expected runtime: 

~3 hours on 1 × GPU A100

---

## Monitoring & Outputs
- Checkpoints with model weights for each stage and resolution ( for example `reecap_s[1-5]_i[16000].pt`) are written under the chosen `OUTDIR` (default: `experiments/` ).
- `train.py` can resume from the latest checkpoint automatically.

---

## Demo & Visualisation
- After training, use `experiments/reconstruct_disease_in_5_effect.py` to create interpolation figures and sample grids.

---

## Tips
- Ensure the AnnData file fits in memory; the supplied configs target the low-memory fundus embeddings. Consider sub-sampling or sharding for larger cohorts.
- If Visdom is unavailable on your cluster, keep the `--no_vis` flag or switch to `--np_vis` to log snapshots without a server.
- Record the git commit and config JSON alongside generated figures for reproducibility.

---

## Attribution
The trainers and model definitions are adapted from [facebookresearch/pytorch_GAN_zoo](https://github.com/facebookresearch/pytorch_GAN_zoo). Please cite the original project alongside the HistoGWAS manuscript when using these artifacts.


