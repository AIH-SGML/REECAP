# generator.py
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as pl
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from models.progressive_gan import ProgressiveGAN as PGAN

class Generator:
    def __init__(self, config, checkpoint, useGPU=True):
        # Load PGAN configuration
        with open(config, 'rb') as file:
            config = json.load(file)

        # Initialize PGAN (assuming PGAN can take x as direct input)
        self.pgan = PGAN(useGPU=useGPU, storeAVG=True, **config)
        self.pgan.load(checkpoint)
        self.netG = self.pgan.netG
        self.device = self.pgan.device

    def forward(self, x, eps=None):
        """
        Forward pass through the generator.
        x: torch.Tensor or np.ndarray of shape (batch_size, 42)
           consisting of 40-dim embedding + 2-dim one-hot vector.
        """

        if eps is None:
            eps = torch.randn(x.shape[0], 512)
        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if isinstance(eps, np.ndarray):
            eps = torch.Tensor(eps)
        x = x.to(self.device)
        eps = eps.to(self.device)
        with torch.no_grad():
            out = self.netG(eps, x).data.cpu()
            out = 0.5 * (out + 1)
            out = torch.clip(out, 0, 1)
        return out

def load_image_torch(path, size):
    if isinstance(path, (list, np.ndarray)):
        return torch.cat([load_image_torch(_, size) for _ in path])
    return pil_to_tensor(Image.open(path).resize((size, size)))[None] / 255.

def torch_imshow(x):
    pl.imshow(x.permute(1, 2, 0))
