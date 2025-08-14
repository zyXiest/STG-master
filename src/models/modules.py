import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
sys.path.append(ROOT.as_posix())

from torch import Tensor
from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projection(nn.Module):
    def __init__(self,
                 inp_dim: int = 512,
                 d_model: int = 512,
    ):
        super(Projection, self).__init__()
        self.proj = nn.Linear(inp_dim, d_model)
        
    def forward(self, inp: Tensor) -> Tensor:
        return self.proj(inp)


class QueryGuidedAttention(nn.Module):
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super(QueryGuidedAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        self.ffn.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, qst_feat, sequence, temporal_mask=None):

        B, T, D = sequence.size()

        q = qst_feat.unsqueeze(1)  # (B, 1, D)

        k = v = sequence  # (B, T, D)

        if temporal_mask is not None:
            attn_mask = ~temporal_mask.bool()  # (B, T)
        else:
            attn_mask = None

        attn_output, _ = self.attn(query=q, key=k, value=v, key_padding_mask=attn_mask)  # (B, 1, D)

        out = self.norm(attn_output + q)  # (B, 1, D)
        out = self.ffn(out)  # (B, 1, D)

        return out.squeeze(1)  # (B, D)