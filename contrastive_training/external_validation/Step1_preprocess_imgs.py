import os
import glob
import sys
import pandas as pd
from EyeQ_preprocess import EyeQ_process_main

original_img_folder = "../data/external_val/external_val_imgs"
csv_path = "../data/external_val/retina_age_40_70.csv"
output_dir = "../data/external_val/processed_crop"

os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(csv_path)
os.makedirs(output_dir, exist_ok=True)
image_paths = glob.glob(os.path.join(original_img_folder, '*.jpg'))
print(f"Total images to process: {len(image_paths)}")
EyeQ_process_main.process(image_paths, output_dir, img_size=(256, 256))