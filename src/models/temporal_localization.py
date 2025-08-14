import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def extract_patchs_from_video(video_frames, selected_indices, max_len=20):
    """
    Args:
        video_frames: Tensor, shape (B, T, D)
        selected_indices: Tensor, shape (B, k)
        max_len: int
    Returns:
        extracted_features: Tensor, shape (B, max_len, D)
        valid_lengths: Tensor, shape (B,)
    """
    B, T, N, D = video_frames.shape 
    max_len = T
    
    extracted_features = torch.zeros(B, max_len, N, D, device=video_frames.device)
    valid_lengths = torch.zeros(B, dtype=torch.long, device=video_frames.device)
    
    for i in range(B):
        current_selected_indices = selected_indices[i]  # (k,)
        current_selected_indices = torch.sort(current_selected_indices).values
        
        selected_features = video_frames[i, current_selected_indices]  
        
        num_selected = selected_features.size(0)
        valid_lengths[i] = min(num_selected, max_len)

        extracted_features[i, :valid_lengths[i]] = selected_features

    return extracted_features, valid_lengths

def extract_frames_from_video(video_frames, selected_indices, max_len=20):

    B, T, D = video_frames.shape 
    max_len = T
    
    extracted_features = torch.zeros(B, max_len, D, device=video_frames.device)
    valid_lengths = torch.zeros(B, dtype=torch.long, device=video_frames.device)
    
    for i in range(B):
        current_selected_indices = selected_indices[i]  # (k,)
        current_selected_indices = torch.sort(current_selected_indices).values
        
        selected_features = video_frames[i, current_selected_indices]  
        
        num_selected = selected_features.size(0)
        valid_lengths[i] = min(num_selected, max_len)

        extracted_features[i, :valid_lengths[i]] = selected_features[:max_len]

    return extracted_features, valid_lengths


class TemporalGraphQA(nn.Module):
    def __init__(self, hidden_dim, similarity_threshold=0.5, k=5):
        """
        :param Q_dim: (L, D)
        :param G_dim: (T, N, D)
        :param similarity_threshold
        """
        super(TemporalGraphQA, self).__init__()
        self.similarity_threshold = similarity_threshold
        self.k = k

        self.linear_transform = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()

        nn.init.kaiming_normal_(self.linear_transform.weight)
        nn.init.constant_(self.linear_transform.bias, 0)

    def cosine_similarity(self, q, g, mask):
        """
        :param q: (B, L, D)
        :param g: (B, T, N, D)
        :param mask: (B, L)
        :return: (B, T, L, N)
        """
        B, L, D = q.shape
        T, N = g.shape[1], g.shape[2]

        q_norm = F.normalize(q, p=2, dim=-1)        # (B, L, D)
        g_norm = F.normalize(g, p=2, dim=-1)        # (B, T, N, D)
        g_flat = g_norm.view(B, T * N, D)           # (B, T*N, D)

        # q_norm @ g_flat^T，q_norm: (B, L, D), g_flat: (B, T*N, D)
        sim = torch.bmm(q_norm, g_flat.transpose(1, 2))  # (B, L, T*N)

        sim = sim.view(B, L, T, N).permute(0, 2, 1, 3)   # (B, T, L, N)

        mask_expanded = mask.unsqueeze(1).unsqueeze(3).float()

        sim = sim * mask_expanded

        return sim

    def compute_attention_weights(self, similarity_matrix, mask):
        """
        :param similarity_matrix: (B x T x L x N)
        :param mask: (B x L)
        :return: (B x T x L x N)
        """
        B, T, L, N = similarity_matrix.shape

        mask_expanded = mask.unsqueeze(1).unsqueeze(3).float()  # (B x 1 x L x 1)

        similarity_matrix = similarity_matrix * mask_expanded + (1 - mask_expanded) * (-1e9)

        attention_weights = F.softmax(similarity_matrix, dim=-1)  

        return attention_weights

    def compute_weighted_graph_representation(self, attention_weights, G, mask):
        """
        :param attention_weights: (B x T x L x N)
        :param G: (B x T x N x D)
        :param mask: (B x L)
        :return: (B x T x L x D)
        """
        B, T, L, N = attention_weights.shape  
        _, _, N_, D = G.shape 

        assert N == N_

        mask_expanded = mask.unsqueeze(1).unsqueeze(3).float()  # (B x 1 x L x 1)

        attention_weights = attention_weights * mask_expanded  # (B x T x L x N)

        sum_weights = attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9) 
        normalized_attention_weights = attention_weights / sum_weights  # (B x T x L x N)

        aggregated_graph = torch.matmul(normalized_attention_weights, G)  # (B x T x L x N) x (B x T x N x D) -> (B x T x L x D)
        aggregated_graph = aggregated_graph.sum(dim=2)  # (B x T x D)
        
        aggregated_graph = self.linear_transform(aggregated_graph)  # (B x T x D)
        aggregated_graph = self.norm(aggregated_graph)
        aggregated_graph = self.activation(aggregated_graph)

        return aggregated_graph

    def compute_match_scores(self, qst, aggregated_graph, mask):
        """
        :param Q: (B x D)
        :param aggregated_graph: (B x T x D)
        :return: (B x T)
        """
        Q_norm = F.normalize(qst, dim=-1)
        weighted_graph_norm = F.normalize(aggregated_graph, dim=-1)

        Q_expanded = Q_norm.unsqueeze(1)
        match_scores = (Q_expanded * weighted_graph_norm).sum(dim=-1)
 
        match_scores = torch.sigmoid(match_scores * 10) 

        return match_scores

    def filter_temporal_frames(self, match_scores):
        """ 
        :param match_scores:  (B x T)
        :param thresholds: (B)
        :return: (B x T) 
        """
        return match_scores > self.similarity_threshold 

    def select_frames_from_scores(self, scores):
        """
        :param scores: (B x T)
        :return: List of tensors
        """
        B, T = scores.shape  # B = batch size, T = number of frames

        initial_mask = (scores > self.similarity_threshold).float()  # (B, T)

        num_selected = initial_mask.sum(dim=1)  # (B,)

        selected_indices = torch.nonzero(initial_mask, as_tuple=False)  # shape: (num_selected, 2)
        
        mask_video_with_fewer_than_k = num_selected < self.k  # (B,)

        topk_indices = torch.topk(scores, self.k, dim=1).indices  # shape: (B, k)

        refined_selected_indices = []

        for i in range(B):
            if mask_video_with_fewer_than_k[i]:  
                refined_selected_indices.append(topk_indices[i])  
            else:
                video_selected_indices = selected_indices[selected_indices[:, 0] == i][:, 1] 
                refined_selected_indices.append(video_selected_indices)
    
        return refined_selected_indices, num_selected
    
    def positive_temporal_proposal(self, match_scores, threshold=0.5, temperature=0.1):
        """
        Args:
            match_scores (Tensor): (B, T) 
            threshold (float): 
            temperature (float):
        """
        B, T = match_scores.shape
        losses = []

        for i in range(B):
            scores = match_scores[i]
            pos_mask = scores >= threshold
            neg_mask = scores < threshold

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                sorted_scores, indices = torch.sort(scores, descending=True)
                half = T // 2
                pos_indices = indices[:half]
                neg_indices = indices[half:]

                pos_scores = scores[pos_indices]
                neg_scores = scores[neg_indices]
            else:
                pos_scores = scores[pos_mask]
                neg_scores = scores[neg_mask]

            logits_pos = pos_scores.unsqueeze(1) / temperature  # (Tp, 1)
            logits_neg = neg_scores.unsqueeze(0) / temperature  # (1, Tn)

            logits = torch.cat([logits_pos, logits_neg.expand(logits_pos.size(0), -1)], dim=1)  # (Tp, 1 + Tn)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=match_scores.device)

            loss = F.cross_entropy(logits, labels)
            losses.append(loss)

        return torch.stack(losses).mean()
    

    def negative_query_contrastive(self, match_scores, features_G, features_Q, threshold=0.5, temperature=0.1):
        """
        Args:
        - match_scores: B x T 
        - features_G: B x T x D 
        - features_Q: B x D 
        """
        B, T, D = features_G.shape

        
        top_idx = match_scores.argmax(dim=1)  # shape: [B]

        batch_indices = torch.arange(B, device=features_G.device)
        Fp = features_G[batch_indices, top_idx]  # shape: [B, D]

        Fp = F.normalize(Fp, dim=1)            # (B, D)
        features_Q = F.normalize(features_Q, dim=1)    # (B, D)
        logits = torch.matmul(Fp, features_Q.T)  # shape: [B, B]
        logits = logits / temperature

        labels = torch.arange(B, device=features_G.device)

        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, Q, G, qst, mask):
        """
        :param Q: (B x L x D)
        :param G:  (B x T x N x D)
        :param mask:  (B x L)
        """

        B, L, D = Q.shape
        T, N, D_ = G.shape[1], G.shape[2], G.shape[3]
        assert D == D_

        similarity_matrix = self.cosine_similarity(Q, G, mask)

        attention_weights = self.compute_attention_weights(similarity_matrix, mask)

        aggregated_graph = self.compute_weighted_graph_representation(attention_weights, G, mask)

        match_scores = self.compute_match_scores(qst, aggregated_graph, mask)

        selected_frames = self.filter_temporal_frames(match_scores)  # (B x T)

        selected_indices, num_selected = self.select_frames_from_scores(match_scores)

        valid_lengths = torch.zeros_like(selected_frames)  # (B x T)
        
        for i, indices in enumerate(selected_indices):
            valid_lengths[i, indices] = 1 

        loss_pt = self.positive_temporal_proposal(match_scores) 

        loss_nq = self.negative_query_contrastive(match_scores, aggregated_graph, qst)

        loss = {'loss_pt': 0.1 * loss_pt , 'loss_nq': 0.1 * loss_nq}
        
        return aggregated_graph, selected_frames, valid_lengths, selected_indices, match_scores, loss
