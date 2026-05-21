import argparse
import torch
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader, TensorDataset
from loss import RnCLoss, RnCLoss_original
from timm.models.layers import trunc_normal_
from util.utils import AverageMeter
from model import VisionTransformer
from tqdm import tqdm
import numpy as np
import pandas as pd
from os.path import join
from functools import partial
import time
import os
import glob

def get_trainer(args, device):
    if args.train_mode == "supervised":
        return SupervisedTrainer(args, device)

    if args.train_mode == "rnc_original":
        return RnCTrainer_original(args, device)

class Trainer:

    def compute_embeddings(self, data_loader):
        self.model.eval()
        self.model.patch_drop_rate = 0
        embeddings = []
        file_paths = []
        with torch.no_grad():
            for X_batch, _, img_paths in data_loader:
                X_batch = X_batch.float()
                Z_batch = self.model.forward_features(X_batch.to(self.device))
                if type(Z_batch) == dict:
                    patch_tokens = Z_batch['x_norm_patchtokens']
                    Z_batch = torch.mean(patch_tokens, dim=1)
                embeddings.append(Z_batch.cpu())
                file_paths.extend(img_paths)
        embeddings = torch.cat(embeddings, 0)
        return embeddings, file_paths
        
    def save_model_and_optimizer(self, filepath):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filepath,
        )

    def resume(self, args):
        if args.resume:
            fnames = glob.glob(join(args.outdir, "model_epoch_*.pth"))
            if len(fnames)==0:
                start_epoch = 0
            else:
                fname = np.sort(fnames)[-1]
                self.load_model_and_optimizer(fname)
                start_epoch = int(fname.split('_')[-1].split('.pth')[0]) + 1
        else:
            start_epoch = 0
        self.start_epoch = start_epoch

    def load_model_and_optimizer(self, fname):
        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])


def get_model(args, device):
 
    d_out = 1 # regression output dimension
    img_size=224
    global_pool=True
    model = VisionTransformer(patch_size=16, 
                                embed_dim=args.zdim, 
                                depth=24, 
                                num_heads=16, 
                                mlp_ratio=4, 
                                qkv_bias=True,
                                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                img_size=img_size,
                                num_classes=d_out,
                                drop_path_rate=args.drop_path_rate,
                                pos_drop_rate=args.pos_drop_rate,
                                patch_drop_rate=args.patch_drop_rate,
                                global_pool=global_pool,
                                mask_ratio=args.mask_ratio)
    
    chkpt_dir = '../data/RETFound_mae_natureCFP.pth'
    checkpoint = torch.load(chkpt_dir, map_location=device, weights_only=False)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    if global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    return model


class SupervisedTrainer(Trainer):
    def __init__(self, args, device):
        self.device = device
        self.verbose = args.verbose
        self.model = get_model(args, device)
        trunc_normal_(self.model.head.weight, std=2e-5)
        self.model.to(device)
        self.loss_fn = MSELoss()

        self.start_epoch = 0
        self.current_epoch = self.start_epoch 
        self.resume(args)
        self.freeze_layers_until_epoch = args.freeze_layers_until_epoch
        if self.freeze_layers_until_epoch:
            self.lr = 0.01
            for name, param in self.model.named_parameters():
                if 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else: 
            self.lr = args.lr
            
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=1e-5)

    def train_model(self, args, train_loader):
        self.model.train()
        total_loss = 0
        iterator = tqdm(train_loader) if self.verbose else train_loader
        self.on_epoch_begin_training() 
        for X_batch, y_batch, file_path in iterator:
            X_batch = X_batch.float().to(self.device)
            y_batch = y_batch.float()[:, None].to(self.device) 
            self.optimizer.zero_grad()
            y_pred = self.model.forward(X_batch)
            loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        return avg_loss

    def on_epoch_begin_training(self):
        self.current_epoch += 1

    def validate_model(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, file_path in val_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float()[:, None].to(self.device)
                y_pred = self.model.forward(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss

    def print_last_layer_info(self):
        last_layer_name = None
        last_layer_parameters = None
        for name, param in self.model.named_parameters():
            last_layer_name = name
            last_layer_parameters = param

class RnCTrainer_original(Trainer):

    def __init__(self, args, device):

        self.device = device
        self.verbose = args.verbose
        self.model = get_model(args, device).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.metric = 'l1'
        self.loss_fn = RnCLoss_original(args.temperature, self.metric)
        self.resume(args)

    def train_model(self, args, train_loader):
        self.model.patch_drop_rate=args.patch_drop_rate
        self.model.train()
        total_loss = 0
        iterator = tqdm(train_loader) if self.verbose else train_loader
        for idx, data_tuple in enumerate(train_loader): 
            images, labels, path = data_tuple
            labels = labels.float()[:,None].to(self.device)
            bsz = labels.shape[0]
            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            features = self.model.forward_features(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = self.loss_fn.forward(features, labels)
            total_loss += loss.item() * bsz
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / len(train_loader.dataset)

        return avg_loss

    def validate_model(self, val_loader):
        self.model.patch_drop_rate = 0
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch, file_path in val_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float()[:,None].to(self.device)
                features = self.model.forward_features(X_batch)
                loss = self.loss_fn.forward(features, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        avg_loss = total_loss / len(val_loader.dataset)
        return avg_loss

