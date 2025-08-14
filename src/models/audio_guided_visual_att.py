import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_mask(lengths, T, N, device='cuda'):
    """
    Args:
        lengths: List[int] 
        T: int 
        N: int 
    Returns:
        mask: Tensor (B, T * N)
    """
    batch_size = len(lengths)

    lengths_tensor = torch.tensor(lengths, device=device)

    mask = torch.arange(T, device=device).expand(batch_size, T) < lengths_tensor.unsqueeze(1)
    
    return mask  

def generate_patch_mask(lengths, T, N, device='cuda'):
    """
    Args:
        lengths: List[int] 
        T: int 
        N: int 
    Returns:
        mask:  (B, T * N)
    """
    batch_size = len(lengths)

    lengths_tensor = torch.tensor(lengths, device=device)

    mask = torch.arange(T * N, device=device).expand(batch_size, T * N) >= lengths_tensor.unsqueeze(1).expand(batch_size, T * N)
    
    return mask  

class AudioGuidedVisualAttn(nn.Module):
    def __init__(self, dim_model=512, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward = 4*dim_model,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads)
    
    def forward(self, audio_feat, visual_feat, valid_lengths):
        """
        Args:
            audio_feat: (B, T, D) 
            visual_feat: (B, T, N, D) 
            mask: (B, T)
        Returns:
            audio_guided_visual: (B, T, D)
        """
        B, T, N, D = visual_feat.shape

        visual_feat_flat = visual_feat.view(B, T * N, D)  # (B, T * N, D)
        visual_feat_flat = visual_feat_flat.permute(1, 0, 2)  # (T * N, B, D)

        audio_feat = audio_feat.permute(1, 0, 2)  # (T, B, D)

        patch_mask = generate_patch_mask(valid_lengths, T, N)

        audio_guided_visual, audio_patch_weights = self.cross_attention(
            query=audio_feat,               # (T, B, D)
            key=visual_feat_flat,           # (T * N, B, D)
            value=visual_feat_flat,         # (T * N, B, D)
            # attn_mask=query_mask,
            key_padding_mask=patch_mask     # (B, T)
        )

        return audio_guided_visual.permute(1, 0, 2), audio_patch_weights  # (B, T, D)
