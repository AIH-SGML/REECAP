# RETFound RnC Training

This is a **contrastive learning framework** designed to **fine-tune the RETFound model encoder** using the **Rank-N-Contrast (RnC) loss**.  
The code is adapted from:  
- [RETFound](https://github.com/rmaphoh/RETFound)  
- [Rank-N-Contrast](https://github.com/kaiwenzha/Rank-N-Contrast)  

---

## Repository Structure
```
RETFound_RnC_training/
├── model.py              # Vision Transformer model
├── loss.py               # Contrastive / regression (MSE) loss functions
├── trainer.py            # Core training logic (Trainer, RnCTrainer, etc.)
├── data.py               # Custom dataset and DataLoader utilities
├── preprocess_imgs.py    # Image preprocessing script
├── train.py              # Main training entry script
├── requirements.txt      # Package requirements
├── contlearn.yml         # Conda environment definition
└── util/                 # Helper utilities
    ├── utils.py          # Metrics and helper functions
    ├── transforms.py     # Image transforms
    └── pos_embed.py      # Positional embedding utilities
```

---

## Usage

### 1. Installation
```
conda env create -f contlearn.yml
conda activate contlearn
```

OR

```
conda create -n contlearn python=3.9.18
conda activate contlearn
conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install \
timm==0.9.16 \
numpy==1.26.4 \
pandas==2.2.1 \
scikit-learn==1.4.1.post1 \
pillow==10.2.0 \
tqdm==4.67.1
```

### 2. Prepare Data

Edit `data/path_table.csv` to point to your image dataset.

Each row should include image paths and their corresponding **age labels** (or other regression targets).

The example toy dataset is already preprocessed and filtered for quality using Reject<=0.8 threshold in EyeQ library. 

For preprocessing of larger datasets, please install our modified version of the
[EyeQ library](https://github.com/HzFu/EyeQ.git):
```
pip install git+https://github.com/AIH-SGML/EyeQ.git
```

> **Note:** This is a fork of the [original EyeQ](https://github.com/HzFu/EyeQ.git)
> with one addition: a function that crops non-square images to square after centralizing.
> This is better suited for the RETFound encoder when the fundus image circle is
> slightly clipped on one side and cropping to square does not lead to losing too much information.
> For UKBB images, where the full fundus circle is typically present, the original EyeQ works equally well.

### Starting Weights for RETFound

Download the RETFound model weights (ViT-Large, MAE pretraining) from the original release:

👉 https://huggingface.co/YukunZhou/RETFound_mae_natureCFP

Download the file:
`RETFound_mae_natureCFP.cfp`

Place it in the `./data` folder.

The pretrained REECAP weights (result of fine-tuning on UK Biobank) can be downloaded from HuggingFace:

👉 https://huggingface.co/LiubovShilova/REECAP

---

### 3. Fine-tune the model to age using RnC loss

Example run training on folds 1-4 and testing on fold 5:
```
python train.py \
    --path_table_file ../data/path_table.csv \
    --train_mode rnc_original \
    --lr 0.00005 \
    --num_epochs 4 \
    --batch_size 4 \
    --aug yes \
    --leave_fold_out 5 \
    --outdir ../data
```
Expected output:

`best_model.pth` - the weights for model with the lowest test loss
`best_train_embeddings.csv` - the corresponding embeddings extracted from the train set
`best_test_embeddings.csv` - the corresponding embeddings extracted from the test set

Expected runtime:

~ 2 mins on 1 GPU A100 