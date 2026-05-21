# Contrastive Training for Feature Extraction

This project is a **contrastive learning framework** designed to **fine-tune the RETFound model encoder** using the **Rank-N-Contrast (RnC) loss**.  
The code is adapted from:  
- [RETFound](https://github.com/rmaphoh/RETFound)  
- [Rank-N-Contrast](https://github.com/kaiwenzha/Rank-N-Contrast)  

---

## Repository Structure
```
contrastive_training/
├── RETFound_RnC_training/         # Fine-tuning code (see README inside)
│   ├── model.py                   # Vision Transformer model
│   ├── loss.py                    # Contrastive / regression (MSE) loss functions
│   ├── trainer.py                 # Core training logic (Trainer, RnCTrainer, etc.)
│   ├── data.py                    # Custom dataset and DataLoader utilities
│   ├── preprocess_imgs.py         # Image preprocessing script
│   ├── train.py                   # Main training entry script
│   ├── requirements.txt           # Package requirements
│   ├── contlearn.yml              # Conda environment definition
│   └── util/                      # Helper utilities
│       ├── utils.py               # Metrics and helper functions
│       ├── transforms.py          # Image transforms
│       └── pos_embed.py           # Positional embedding utilities
├── create_adata_with_embeddings/  # PCA reduction and AnnData annotation scripts
│   ├── Step1_pca_embeddings.py
│   └── Step2_annotate_embeddings.py
├── external_validation/           # External cohort validation scripts
│   ├── Step0_download_data.py
│   ├── Step1_preprocess_imgs.py
│   ├── Step2_create_table_paths_phenos.py
│   ├── Step3_extract_features.py
│   ├── Step4_linearprobing.py
│   └── Step5_plot_comparison.py
└── data/                          # Example images and path table
    ├── path_table.csv
    └── imgs/
```

---

## Usage

See ./RETFound_RnC_training/README.md for usage details.