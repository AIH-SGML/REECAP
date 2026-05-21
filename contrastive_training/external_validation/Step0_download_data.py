from datasets import load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
import os

# Download data from https://huggingface.co/datasets/ramankamran/retina-age-analysis

ds = load_dataset("ramankamran/retina-age-analysis")
external_val_dir = "../data/external_val"
IMG_DIR = "..../data/external_val/external_val_imgs"
os.makedirs(external_val_dir, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

ds_all = concatenate_datasets(
    [ds["train"], ds["validation"], ds["test"]]
)

age_min = 40
age_max = 70

ds_age_match_to_ukbb = ds_all.filter(
        lambda x: age_min <= x["patient_age"] <= age_max
    )

ds_age_match_to_ukbb.to_csv("../data/external_val/retina_age_40_70.csv", index=False)