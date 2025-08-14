
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerTemporalGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MultiLayerTemporalGraph, self).__init__()
        self.num_layers = num_layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TemporalGraphLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.input_layer.apply(self.init_weight)
        self.output_layer.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_features, audio_features):
        """
        Args:
            visual_features: (batch_size, T, N, D) 32 60 49 512
            audio_features: (batch_size, T, D_audio) 32 60 512
        Returns:
            visual_output: (batch_size, T, N, output_dim)  512
            audio_output: (batch_size, T, 1, output_dim)   512
        """
        batch_size, T, N, D = visual_features.shape
        
        audio_node = audio_features.unsqueeze(2)  # shape: (batch_size, T, 1, D)
        node_features = torch.cat([visual_features, audio_node], dim=2)  # shape: (batch_size, T, N+1, D)
        
        node_features = self.input_layer(node_features)  # shape: (batch_size, T, N+1, hidden_dim)

        for layer in self.layers:
            node_features, attention_weights, temporal_weights = layer(node_features)
        
        output_nodes = self.output_layer(node_features)  # shape: (batch_size, T, N+1, output_dim)
        
        # visual_output = output_nodes[:, :, :-1, :]  # shape: (batch_size, T, N, output_dim)
        # audio_output = output_nodes[:, :, -1, :]  # shape: (batch_size, T, 1, output_dim)
        
        return output_nodes, attention_weights, temporal_weights


class TemporalGraphLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalGraphLayer, self).__init__()
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=0.1)

        self.intra_visual_thredhold = 0.021  
        self.audio_visual_thredhold = 0.032 

        self.W_c.apply(self.init_weight)
        self.W_s.apply(self.init_weight)
        self.output_proj.apply(self.init_weight)
    
    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def compute_attention(self, Q, K, mask):

        attention_scores = torch.einsum('btih,btjh->btij', Q, K)  # (batch_size, T, N+1, N+1)
        
        attention_scores[mask] = -1e9 
        
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True)[0]
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights 

    def forward(self, node_features):
        """
        Args:
            node_features: (batch_size, T, N+1, hidden_dim)
        Returns:
            temporal_output: (batch_size, T, N+1, hidden_dim)
        """
        batch_size, T, N_plus_1, hidden_dim = node_features.shape 
        
        ############ Step 1: Spatial Update ########### 
        Q = node_features  # (batch_size, T, N+1, hidden_dim)
        K = node_features  # (batch_size, T, N+1, hidden_dim)
        V = node_features  # (batch_size, T, N+1, hidden_dim)
        
        Q = F.normalize(Q, dim=-1)  # 归一化 Q 和 K
        K = F.normalize(K, dim=-1)
        
        mask = torch.zeros(batch_size, T, N_plus_1, N_plus_1, dtype=torch.bool, device=node_features.device)
        for t in range(T):
            mask[:, t, :, :] = torch.eye(N_plus_1, device=node_features.device).bool()
        
        attention_weights = self.compute_attention(Q, K, mask)
        attention_mask=torch.zeros_like(attention_weights, dtype=torch.bool, device=node_features.device)
        attention_mask[:,:,:-1, :-1] = attention_weights[:,:,:-1,:-1] > self.intra_visual_thredhold 
        attention_mask[:,:,-1, :-1] = attention_weights[:,:,-1,:-1] > self.audio_visual_thredhold 
        attention_mask[:,:,:-1, -1] = attention_weights[:,:,:-1,-1] > self.audio_visual_thredhold  
        attention_weights=attention_mask*attention_weights 

        spatial_features = torch.einsum('btij,btjh->btih', attention_weights, V)  # (batch_size, T, N+1, hidden_dim)
        spatial_features = self.activation(self.norm(spatial_features + node_features)) 

        ############ Step 2: Temporal Update ##########
        temporal_nodes = spatial_features.clone().detach()
        current_nodes = spatial_features[:, 1:, :, :]     
        previous_nodes = spatial_features[:, :-1, :, :]    

        temporal_scores = F.cosine_similarity(temporal_nodes[:, 1:, :, :] , temporal_nodes[:, :-1, :, :], dim=-1).unsqueeze(-1)  # shape: [B, T-1, N+1, 1]

        temporal_c = temporal_scores * previous_nodes  # (batch_size, T-1, N+1, hidden_dim)
        temporal_c = self.activation(self.W_c(temporal_c))
        temporal_s = current_nodes * (1 - temporal_scores)
        temporal_s = self.activation(self.W_s(temporal_s))

        # update temporal_features
        temporal_updated = current_nodes + temporal_c + temporal_s  # [B, T-1, N+1, d]

        temporal_output = torch.cat([spatial_features[:, :1, :, :], temporal_updated], dim=1) # [B, T, N+1, d_out]
        temporal_output = self.dropout(temporal_output)
        temporal_output = self.norm(temporal_output)
        temporal_output = self.output_proj(temporal_output) 
        temporal_output = node_features + self.activation(temporal_output) 

        return temporal_output, attention_weights, temporal_scores


class MultiLayerDirectedGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MultiLayerDirectedGraph, self).__init__()
        self.num_layers = num_layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            DirectedGraphLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.input_layer.apply(self.init_weight)
        self.output_layer.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_features, audio_features):

        batch_size, T, N, D = visual_features.shape
        
        audio_node = audio_features.unsqueeze(2)  
        node_features = torch.cat([visual_features, audio_node], dim=2) 
        
        node_features = self.input_layer(node_features) 

        for layer in self.layers:
            node_features, attention_weights, temporal_weights = layer(node_features)
        
        output_nodes = self.output_layer(node_features) 
        
        return output_nodes, attention_weights, temporal_weights


class DirectedGraphLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(DirectedGraphLayer, self).__init__()
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ReLU()
        self.edge_type_embedding = nn.Parameter(torch.randn(3, hidden_dim))  # 0: intra-visual, 1: visual-audio, 2: audio-visual

        self.intra_visual_thredhold = 0.021 
        self.visual_audio_thredhold = 0.032 
        self.audio_visual_thredhold = 0.032 

        self.W_c = nn.Linear(hidden_dim, hidden_dim)
        self.W_s = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(p=0.1)

        self.W_c.apply(self.init_weight)
        self.W_s.apply(self.init_weight)
        self.output_proj.apply(self.init_weight)
    
    def init_weight(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def compute_attention(self, Q, K, mask):

        attention_scores = torch.einsum('btih,btjh->btij', Q, K)  # (batch_size, T, N+1, N+1)
        
        attention_scores[mask] = -1e9 
        
        attention_scores = attention_scores - attention_scores.max(dim=-1, keepdim=True)[0]
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights 

    def forward(self, node_features):
        """
        Args:
            node_features: (batch_size, T, N+1, hidden_dim)
        Returns:
            temporal_output: (batch_size, T, N+1, hidden_dim)
        """
        batch_size, T, N_plus_1, hidden_dim = node_features.shape  ##49+1

        # intral visual attention
        Q_intral_visual = node_features[:,:,:-1, :]  # (batch_size, T, N, hidden_dim)
        K_intral_visual = node_features[:,:,:-1, :]+self.edge_type_embedding[0].unsqueeze(0).unsqueeze(0).unsqueeze(0) # (batch_size, T, N, hidden_dim)
        V_intral_visual = K_intral_visual.clone() 
        Q_intral_visual= F.normalize(Q_intral_visual, dim=-1) 
        K_intral_visual = F.normalize(K_intral_visual, dim=-1)

        intral_visual_attention_scores = torch.einsum('btih,btjh->btij', Q_intral_visual, K_intral_visual)  # (batch_size, T, N, N)

        #get mask
        W = 7
        N = N_plus_1 - 1
        idx = torch.arange(N)
        qr = idx // W  # [N]
        qc = idx % W  # [N]

        r0 = (qr - 1).clamp(0, T - 3)
        c0 = (qc - 1).clamp(0, W - 3) 

        kr = qr[None, :]  # [1, N]
        kc = qc[None, :]  # [1, N]

        r0_ = r0[:, None]  # [N, 1]
        c0_ = c0[:, None]  # [N, 1]

        in_row = (kr >= r0_) & (kr <= r0_ + 2)  # [N, N]
        in_col = (kc >= c0_) & (kc <= c0_ + 2)  # [N, N]
        allow = in_row & in_col  # [N, N]

        self_mask = (qr[:, None] == kr) & (qc[:, None] == kc)  # [N, N]
        allow = allow & (~self_mask)  # [N, N]

        disallow = ~allow  # [N, N]
        disallow = disallow.view(1, 1, N, N).expand(batch_size, T, N, N).to(node_features.device)
        intral_visual_attention_scores[disallow] = -1e9 

        # audio visual attention
        Q_audio_visual = node_features[:,:,-1, :].unsqueeze(2)  # (batch_size, T, 1, hidden_dim)
        K_audio_visual = node_features[:,:,:-1, :]+self.edge_type_embedding[1].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (batch_size, T, N, hidden_dim)
        V_audio_visual = K_audio_visual.clone()  # (batch_size, T, N, hidden_dim)
        Q_audio_visual = F.normalize(Q_audio_visual, dim=-1) 
        K_audio_visual = F.normalize(K_audio_visual, dim=-1)
        audio_visual_attention_scores = torch.einsum('btih,btjh->btij', Q_audio_visual, K_audio_visual)  # (batch_size, T, 1, N)

        # visual audio attention
        Q_visual_audio = node_features 
        K_visual_audio = (node_features[:,:,-1, :]).unsqueeze(2)+self.edge_type_embedding[2].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (batch_size, T, 1, hidden_dim)
        V_visual_audio = K_visual_audio.clone()  # (batch_size, T, 1, hidden_dim)
        Q_visual_audio = F.normalize(Q_visual_audio, dim=-1)  
        K_visual_audio = F.normalize(K_visual_audio, dim=-1)
        visual_audio_attention_scores = torch.einsum('btih,btjh->btij', Q_visual_audio, K_visual_audio)  # (batch_size, T, N, 1)

        total_attention_scores = torch.cat((torch.cat((intral_visual_attention_scores, audio_visual_attention_scores),dim=2), visual_audio_attention_scores), dim=-1)  # (batch_size, T, N+1, N+1)
        mask = torch.zeros(batch_size, T, N_plus_1, N_plus_1, dtype=torch.bool, device=node_features.device)
        for t in range(T):
            mask[:, t, :, :] = torch.eye(N_plus_1, device=node_features.device).bool()

        total_attention_scores[mask] = -1e9 

        total_attention_scores = total_attention_scores - total_attention_scores.max(dim=-1, keepdim=True)[0]
        attention_weights = F.softmax(total_attention_scores, dim=-1)  # (batch_size, T, N+1, N+1)

        intral_visual_attention_scores=attention_weights[:, :, :-1, :-1]  # (batch_size, T, N, N)
        audio_visual_attention_scores = attention_weights[:, :, -1, :-1].unsqueeze(2)  # (batch_size, T, 1, N)
        visual_audio_attention_scores = attention_weights[:, :, :, -1].unsqueeze(-1)  # (batch_size, T, N, 1)
        visual_aggreation = torch.einsum('btij,btjh->btih', intral_visual_attention_scores, V_intral_visual)  # (batch_size, T, N, hidden_dim)
        audio_aggreation = torch.einsum('btij,btjh->btih', audio_visual_attention_scores, V_audio_visual)  # (batch_size, T, 1, hidden_dim)
        visual_audio_aggreation = torch.einsum('btij,btjh->btih', visual_audio_attention_scores, V_visual_audio)  # (batch_size, T, N, hidden_dim)
        spatial_features = torch.cat((visual_aggreation, audio_aggreation), dim=2)  # (batch_size, T, N+1, hidden_dim)
        spatial_features= spatial_features + visual_audio_aggreation  # (batch_size, T, N+1, hidden_dim)
        
        spatial_features = self.activation(self.norm(spatial_features + node_features)) 

        # Temporal Update 
        temporal_nodes = spatial_features.clone().detach()
        current_nodes = spatial_features[:, 1:, :, :]     
        previous_nodes = spatial_features[:, :-1, :, :]    
        temporal_scores = F.cosine_similarity(temporal_nodes[:, 1:, :, :] , temporal_nodes[:, :-1, :, :], dim=-1).unsqueeze(-1)  # shape: [B, T-1, N+1, 1]

        temporal_c = temporal_scores * previous_nodes  # (batch_size, T-1, N+1, hidden_dim)
        temporal_c = self.activation(self.W_c(temporal_c))
        temporal_s = current_nodes * (1 - temporal_scores)
        temporal_s = self.activation(self.W_s(temporal_s))

        temporal_updated = current_nodes + temporal_c + temporal_s  # [B, T-1, N+1, d]

        temporal_output = torch.cat([spatial_features[:, :1, :, :], temporal_updated], dim=1) # [B, T, N+1, d_out]
        temporal_output = self.dropout(temporal_output)
        temporal_output = self.norm(temporal_output)
        temporal_output = self.output_proj(temporal_output) 
        temporal_output = node_features + self.activation(temporal_output) 

        return temporal_output, attention_weights, temporal_scores
