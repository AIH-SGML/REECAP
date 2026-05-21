from torch.utils.data import Dataset
from PIL import Image
import anndata
import torch
import torchvision
import pdb
import numpy as np


class AnnDataset(Dataset):
    def __init__(self, pathDB, transform):

        self.adata = anndata.read_h5ad(pathDB)
        self.condition = self.adata.obs[['eye_left', 'eye_right']].values
        self.transform = transform
        self.return_attrib = False
        
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.adata.X[idx])
        c = torch.from_numpy(self.condition[idx])
        x = torch.cat([x, c], axis=0)
        img_path = self.adata.obs['path'].iloc[idx]
        img = self.transform(Image.open(img_path))
        #x = torch.tensor(x, dtype=torch.float32)
        x = x.clone().detach().to(dtype=torch.float32)
        return img, x


if __name__=='__main__':

    dataset = AnnDataset(pathDB,  transform)
    img, x = dataset.__getitem__(0)
