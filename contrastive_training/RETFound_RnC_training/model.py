import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Iterable, List#, Optional, cast
from torch import Tensor
import numpy as np
import typing
import math
from functools import partial
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

#######from models_vit in RETFound##########

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ 
    Vision Transformer with support for global average pooling. 
    Code for RETFound implementation: https://github.com/rmaphoh/RETFound_MAE/blob/main/models_vit.py
    Code for timm implementation: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L748
    In timm implementation the global pooling is applied at the forward_head, in RETFound it is applied in forward_features. However refound also does not overwrite forward_head, so it seems that the global_pooling would be then applied twice.
    """
    def __init__(self, global_pool=True, mask_ratio=0, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.mask_ratio=mask_ratio
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device) 
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_encoder(self, x, mask_ratio):
       
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:

        x = self.fc_norm(x)
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
