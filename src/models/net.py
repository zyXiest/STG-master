import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict
from src.models.audio_guided_visual_att import AudioGuidedVisualAttn

from src.models.modules import Projection
from src.models.Audio_Visual_ST import *
from src.models.temporal_localization import *
from src.models.Audio_Visual_ST import *
import copy

class GlobalHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(GlobalHanLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GlobalSelfAttn(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(GlobalSelfAttn, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_v, mask=None, src_key_padding_mask=None):
        
        visual_output = src_v

        for i in range(self.num_layers):
            visual_output = self.layers[i](src_v, src_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            visual_output = self.norm2(visual_output)

        return visual_output
    

class STG(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 video_dim: int = 512,
                 patch_dim: int = 768,
                 audio_dim: int = 128,
                 graph_type: str = 'STG',
                 **kwargs
    ):
        super(STG, self).__init__()
        
        self.audio_proj = Projection(audio_dim, d_model)
        self.video_proj = Projection(video_dim, d_model)
        self.patch_proj = Projection(patch_dim, d_model)
        self.words_proj = Projection(video_dim, d_model)
        self.quest_proj = Projection(video_dim, d_model)

        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(d_model, 42)
        
        if graph_type == 'MTG':
            self.av_gnn = MultiLayerDirectedGraph(input_dim=d_model, hidden_dim=d_model, output_dim=d_model)
        else:
            self.av_gnn = MultiLayerTemporalGraph(input_dim=d_model, hidden_dim=d_model, output_dim=d_model)

        self.TempSegsSelect_Module = TemporalGraphQA(hidden_dim=d_model, similarity_threshold=0.5)
        self.AudioGuidedVisualAttn = AudioGuidedVisualAttn()
        self.GlobalSelf_Module = GlobalSelfAttn(GlobalHanLayer(d_model=d_model, nhead=8, 
                                                               dim_feedforward=d_model), 
                                                               num_layers=1)
        
        self.audio_proj.apply(self.init_weight)
        self.video_proj.apply(self.init_weight)
        self.words_proj.apply(self.init_weight)
        self.quest_proj.apply(self.init_weight)
        self.patch_proj.apply(self.init_weight)
        self.head.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def sub_forward(self, 
                    reshaped_data: Dict[str, Tensor],
                    prefix: str = ''
    ):
        
        quest = reshaped_data[f'{prefix}quest'].squeeze(-2) 
        audio = reshaped_data[f'{prefix}audio']
        video = reshaped_data[f'{prefix}video']
        words = reshaped_data[f'{prefix}word'].squeeze(-3)
        words_mask = reshaped_data[f'{prefix}word_mask']
        patch = reshaped_data[f'{prefix}patch'] if f'{prefix}patch' in reshaped_data else None
        
        return quest, words, words_mask, audio, video, patch
    
    
    def forward(self, reshaped_data: Dict[str, Tensor]):
        '''
            input audio shape:      [B, T, AC]
            input pos_frames shape: [B, T, VC, FH, FW]
            input question shape:   [B, D]
            input neg_frames shape: [B, T, VC, FH, FW]
        '''
        return_dict = {}

        quest, words, words_mask, audio, video, patch =self.sub_forward(reshaped_data, prefix='')
        B, T, D = video.shape
            
        # Projection
        audio = self.audio_proj(audio) # [B, T, D]
        video = self.video_proj(video) # [B, T, D]
        words = self.words_proj(words) # [B, 77, D]
        quest = self.quest_proj(quest) # [B, D]
        patch = self.patch_proj(patch) # [B, T, P, D]
        
        # ################## Audio Visual Spatial-Temporal Graph ####################
        output_nodes, attention_weights, temporal_weights = self.av_gnn(patch, audio)
        # patch_node = output_nodes[:, :, :-1, :]  # shape: (batch_size, T, N, output_dim)
        # audio_node = output_nodes[:, :, -1, :]  # shape: (batch_size, T, 1, output_dim)

        # #################### Adaptive Temporal Localization ####################
        aggregated_graph, selected_frames, valid_lengths, selected_indices, match_scores, temp_loss = self.TempSegsSelect_Module(words, output_nodes, quest, words_mask)
        selected_visual, valid_lengths = extract_frames_from_video(video, selected_indices)
        selected_audio, valid_lengths = extract_frames_from_video(audio, selected_indices)
        selected_nodes, valid_lengths = extract_frames_from_video(aggregated_graph, selected_indices)
        
        ###  Question Fusion module **************************************************************************    

        # -------> Concat with T-dim
        fusion_feat = torch.cat((audio, video, selected_audio, selected_visual, selected_nodes, words), dim=1)
        av_fusion_feat = self.GlobalSelf_Module(fusion_feat)
    
        fusion = av_fusion_feat.mean(dim=-2)

        fusion = torch.mul(fusion, quest)  
        fusion = F.tanh(fusion)

        output = self.head(fusion)
        return_dict.update({'out': output, 'loss_pt': temp_loss['loss_pt'], 'loss_nq': temp_loss['loss_nq']})
        return return_dict
