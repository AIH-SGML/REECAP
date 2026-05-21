import os
from os.path import join
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from utils import TwoCropTransform
from functools import partial
import torch.nn as nn
import pdb

from model import VisionTransformer  

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['Path']
        age = self.dataframe.iloc[idx]['Age']
        
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image, age, img_path
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None, None, None

def get_model(args, device, checkpoint_path=None):
    if args['model'] == 'retfound':
        d_out = args['d_out'] if 'd_out' in args else 1
        img_size = 224
        global_pool = True
        model = VisionTransformer(patch_size=16, 
                                  embed_dim=args['zdim'], 
                                  depth=24, 
                                  num_heads=16, 
                                  mlp_ratio=4, 
                                  qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                  img_size=img_size,
                                  num_classes=d_out,
                                  drop_path_rate=args['drop_path_rate'],
                                  pos_drop_rate=args['pos_drop_rate'],
                                  patch_drop_rate=args['patch_drop_rate'],
                                  global_pool=global_pool,
                                  mask_ratio=args['mask_ratio'])
        
    
        checkpoint = torch.load(checkpoint_path, map_location=device)
        msg = model.load_state_dict(checkpoint, strict=False)
        #if global_pool:
            #assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        return model
    else:
        raise ValueError("Unsupported model type")

class Trainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)

    def compute_embeddings(self, data_loader):
        self.model.eval()
        embeddings = []
        file_paths = []
        with torch.no_grad():
            for X_batch, _, img_paths in data_loader:
                X_batch = X_batch.float().to(self.device)
                Z_batch = self.model.forward_features(X_batch)
                if isinstance(Z_batch, dict):
                    patch_tokens = Z_batch['x_norm_patchtokens']
                    Z_batch = torch.mean(patch_tokens, dim=1)
                embeddings.append(Z_batch.cpu())
                file_paths.extend(img_paths)
        embeddings = torch.cat(embeddings, 0)
        return embeddings, file_paths

def save_embeddings_to_csv(file_paths, embeddings, output_dir, filename):
    
    # Create DataFrame with file paths
    df = pd.DataFrame({'file_path': file_paths})
    embeddings_df = pd.DataFrame(embeddings, columns=[f'embedding_{i+1}' for i in range(embeddings.shape[1])])  
    df = pd.concat([df, embeddings_df], axis=1)
    output_path = join(output_dir, filename)
    df.to_csv(output_path, index=False)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
args = {
    'model': 'retfound',
    'zdim': 1024,  
    'drop_path_rate': 0.1,
    'pos_drop_rate': 0.0,
    'patch_drop_rate': 0.0,
    'mask_ratio': 0.0,
    'd_out': 1  
}

script_dir = os.path.dirname(os.path.abspath(__file__))

path_table_file = os.path.join(script_dir, '../data/external_val/retina_age_40_70.csv')
output_folders = {
    'REECAP': os.path.join(script_dir, '../data/external_val/processed/REECAP_features'),
    'RETFound': os.path.join(script_dir, '../data/external_val/RETFound_features')
}
checkpoint_paths = {
    'REECAP': os.path.join(script_dir, '../data/REECAP.pth'),
    'RETFound': os.path.join(script_dir, '../data/RETFound_mae_natureCFP.pth')
}

df = pd.read_csv(path_table_file)

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.Lambda(lambda x: np.array(x) / 255.0),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
                                transforms.Lambda(lambda x: x.float())
                               ])
for model_name, checkpoint_path in checkpoint_paths.items():
    output_folder = output_folders[model_name]
    os.makedirs(output_folder, exist_ok=True)
    model = get_model(args, device, checkpoint_path)
    dataset = CustomDataset(df, transform=transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    trainer = Trainer(model, device)
    embeddings, file_paths = trainer.compute_embeddings(data_loader)
    save_embeddings_to_csv(file_paths, embeddings, output_folder, 'embeddings.csv')