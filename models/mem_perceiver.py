from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder

def attend(query, key, value, dropout_prob=0.0, train=True):
    r"""
        input: query, key, value
        return: attention outputs and attention scores
    """
    
    _, _, _, head_dim = query.shape
    
    attention = F.softmax(torch.einsum("bhqd,bhkd->bhqk", query, key) / (head_dim**(0.5)), dim=-1) 
    attention = F.dropout(attention, p=dropout_prob, training=train)
    weighted_sum = torch.einsum("bhqk,bhkd->bhqd", attention, value) # (batch_size, num_heads, query_len, head_dim)

    return weighted_sum, attention


class MLP(nn.Module):
    """A dense module following attention in Transformer block."""
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float,
        dropout_prob: float = 0.0,
    ):
        super(MLP, self).__init__()
        inner_dim = int(hidden_dim*mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout_prob)
        self.norm = nn.LayerNorm(inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout_prob)

    def reset_parameters(self):
        self.fc2.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x        


class Attention(nn.Module):
    """{Cross, Self}-Attention Module"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        input_qdim: int = None,
        input_kdim: int = None,
        dropout_prob: float = 0.0,
    ): 
        super(Attention, self).__init__()
        if input_qdim is None:
            input_qdim = hidden_dim
        if input_kdim is None:
            input_kdim = input_qdim

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads 
        self.dropout_prob = dropout_prob

        self._query = nn.Linear(input_qdim, hidden_dim)
        self._key = nn.Linear(input_kdim, hidden_dim)
        self._value = nn.Linear(input_kdim, hidden_dim)  
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        self._query.reset_parameters()
        self._key.reset_parameters()
        self._value.reset_parameters()
        self.proj.reset_parameters()
        
    def forward(self, query, key=None, value=None):
        if key is None:
            # self-attention
            key = query
            value = query
        else:
            # cross-attention
            if value is None:
                value = key
        
        B, N, D = query.shape
        H = self.num_heads
        head_dim = self.head_dim

        # (batch_size, query_len, hidden_dim) -> (batch_size, num_heads, query_len, head_dim)
        Q = self._query(query).reshape(B, -1, H, head_dim).transpose(1,2)
        K = self._key(key).reshape(B, -1, H, head_dim).transpose(1,2)
        V = self._value(value).reshape(B, -1, H, head_dim).transpose(1,2)

        # (batch_size, num_heads, query_len, head_dim) -> (batch_size, query_len, hidden_dim)
        attn_output, attn_output_weights = attend(Q, K, V, self.dropout_prob, self.training) 
        out = attn_output.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.proj(out)

        return out, attn_output_weights

class SelfAttention(nn.Module):
    """Self Attention Module including Normalization, Dropout, MLP"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        input_dim: int = None,
        dropout_prob: float = 0.0,
        num_experts: int = None,
        moe_ratio: float = None, 
    ):
        super(SelfAttention, self).__init__()
        if input_dim is None:
            input_dim = hidden_dim
        self.num_experts = num_experts

        self.attention = Attention(hidden_dim, num_heads, input_dim, input_dim, dropout_prob)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout_prob)    
        self.norm2 = nn.LayerNorm(hidden_dim)

        if num_experts is not None:
            self.moe = nn.ModuleList([
                MLP(hidden_dim, moe_ratio, dropout_prob)
                for _ in range(num_experts)
            ])

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.mlp.reset_parameters()

        if self.num_experts is not None:
            for mlp in self.moe:
                mlp.reset_parameters()

    def forward(self, x, expert_idx=None):
        attn_outputs, _ = self.attention(self.norm1(x))
        x = x + attn_outputs
        x = x + self.mlp(self.norm2(x))
        if expert_idx is not None:
            x = x + self.moe[expert_idx](self.norm2(x))
        return x


class CrossAttention(nn.Module):
    """Cross Attention Module including Normalization, Dropout, MLP"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        input_qdim: int = None,
        dropout_prob: float = 0.0,
        num_experts: int = None,
        moe_ratio: float = None, 
    ):
        super(CrossAttention, self).__init__()
        self.num_experts = num_experts
        if input_qdim is None:
            input_qdim = hidden_dim

        self.attention = Attention(hidden_dim, num_heads, input_qdim, input_qdim, dropout_prob)
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout_prob)    

        self.q_norm = nn.LayerNorm(input_qdim)
        self.kv_norm = nn.LayerNorm(input_qdim)
        self.mlp_norm = nn.LayerNorm(hidden_dim)

        if num_experts is not None:
            self.moe = nn.ModuleList([
                MLP(hidden_dim, moe_ratio, dropout_prob)
                for _ in range(num_experts)
            ])

    def reset_parameters(self):
        self.attention.reset_parameters()
        self.mlp.reset_parameters()

        if self.num_experts is not None:
            for mlp in self.moe:
                mlp.reset_parameters()

    def forward(self, query, key, expert_idx=None, need_weights=False):
        attn_output, attn_output_weights = self.attention(self.q_norm(query), self.kv_norm(key))
        x = query + attn_output
        x = x + self.mlp(self.mlp_norm(x))

        if expert_idx is not None:
            x = x + self.moe[expert_idx](self.mlp_norm(x))
        
        if need_weights:
            return x, attn_output_weights
        else: 
            return x

class MemPerceiver(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_samples: int,
        num_heads: int,
        num_layers: int,
        num_latents: int,
        hidden_dim: int,
        mlp_ratio: float,
        dropout_prob: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        top_k: int = 5,
        attn_retrival: bool = False,
        ensemble: bool = False,
        is_cos_sim: bool = True,
    ):
        super(MemPerceiver, self).__init__()

        self.top_k = top_k
        self.attn_retrival = attn_retrival
        self.ensemble = ensemble 
        self.is_cos_sim = is_cos_sim
        self.hidden_dim = hidden_dim
        self.num_features = self.calculate_num_features(col_names_dict)
        
        self.register_buffer('memory', torch.empty(num_samples, num_latents, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_features, hidden_dim))
        self.latents = nn.Parameter(torch.empty(1, num_latents, hidden_dim))
        self.query = nn.Parameter(torch.empty(1, 1, hidden_dim))

        self.tensor_frame_encoder = StypeWiseFeatureEncoder(
            out_channels=hidden_dim,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }
        )
        self.encoder = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
        )
        self.blocks = nn.Sequential(
            *[
                SelfAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_prob=dropout_prob,
                ) 
                for _ in range(num_layers)
            ]
        )
        self.decoder = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )
        self.reset_parameters()

    def calculate_num_features(self, col_names_dict):
        num_features = 0
        for k, v in col_names_dict.items():
            num_features += len(v)
        return num_features   

    def reset_parameters(self):
        nn.init.normal_(self.memory)
        nn.init.normal_(self.pos_embedding)
        nn.init.trunc_normal_(self.latents, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

        self.tensor_frame_encoder.reset_parameters()
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()

        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()

    @torch.no_grad()
    def retrieve(self, x, attn_output_weights=None, is_cos_sim=True):
        r"""
            Retrieve top k instances from memory bank.
            input: 
                x: Encoded Latents (B, L, D)
                attention: Attention Scores from Encoder (B, L, F) 
                    where L is the number of latents, and F is the number of features.
            return: 
                tensor (batch_size, k, num_latents, hidden_dim)
        """
        batch_size, num_latents, hidden_dim = x.shape
        num_samples = self.memory.shape[0]

        if is_cos_sim:
            x_norm = F.normalize(x, dim=-1)  # (B, L, D)
            memory_norm = F.normalize(self.memory, dim=-1)  # (N, L, D)

            cos_sim = torch.einsum("bld,nld->bnl", x_norm, memory_norm)  # (B, N, L)

            if attn_output_weights is None:
                # option 1: average cosine similarity on latent dimension
                cos_sim_avg = cos_sim.mean(dim=-1)  # (B, N)
            else:
                # option 2: average the cos sim with attention score
                latent_weights = attn_output_weights.mean(dim=-1) # (B, L)
                cos_sim_avg = torch.einsum("bnl,bl->bn", cos_sim, latent_weights) # (B, N)

            # retrieve top k for each batch
            _, top_k_indices = torch.topk(cos_sim_avg, k=min(self.top_k, num_samples), dim=-1) # (B, K)
        else:
            x_expanded = x.unsqueeze(1)                # (B, 1, L, D)
            memory_expanded = self.memory.unsqueeze(0) # (1, N, L, D)

            # Compute squared L2 distance for each latent
            l2_dist_squared = ((x_expanded - memory_expanded) ** 2).sum(dim=-1)  # (B, N, L)

            if attn_output_weights is None:
                # option 1: average distance across latent dimension
                dist_avg = l2_dist_squared.mean(dim=-1)  # (B, N)
            else:
                # option 2: weighted average using attention scores
                latent_weights = attn_output_weights.mean(dim=-1)  # (B, L)
                dist_avg = torch.einsum("bnl,bl->bn", l2_dist_squared, latent_weights)  # (B, N)

            # retrieve top-k closest(smallest distances)
            _, top_k_indices = torch.topk(-dist_avg, k=min(self.top_k, num_samples), dim=-1)  # (B, K)
    

        retrievals = self.memory[top_k_indices.view(-1), :, :]  # (B*k, L, D)
        retrievals = retrievals.view(batch_size, self.top_k, num_latents, hidden_dim)  # (B, k, L, D)        

        return retrievals

    @torch.no_grad()
    def update_memory(self, dataloader):
        self.eval()
        all_indices = []
        all_representations = []

        for tf, index in dataloader:
            batch_size = len(tf)
            latents = self.latents.repeat(batch_size, 1, 1)

            tf = tf.to(self.memory.device)
            index = index.to(self.memory.device)

            # encode into latent
            x, _ = self.tensor_frame_encoder(tf)       
            x = x + self.pos_embedding                 
            x = self.encoder(latents, x) # (batch_size, num_latents, hidden_dim)
            
            all_indices.append(index)
            all_representations.append(x.detach())     

        idx = torch.cat(all_indices, dim=0)            
        feat = torch.cat(all_representations, dim=0)   
        self.memory.index_copy_(0, idx, feat)          

    def forward(self, tf):
        is_cos_sim = self.is_cos_sim
        # (B, F, 1) -> (B, F, D)
        if not self.ensemble:
            batch_size = len(tf)
            latents = self.latents.repeat(batch_size, 1, 1)
            query = self.query.repeat(batch_size, 1, 1)

            # Embed features and encode into latent space
            x, _ = self.tensor_frame_encoder(tf) # x, all_col_names
            x = x + self.pos_embedding
            
            if self.attn_retrival:
                # Retrieve top-k from memory bank with weighted sum using attention score
                x, attn_output_weights = self.encoder(latents, x, need_weights=True) # (B, L, D)
                retrievals = self.retrieve(x, attn_output_weights=attn_output_weights, is_cos_sim=is_cos_sim).view(batch_size, -1, x.shape[-1]) # (B, K*L, D)

            else:
                # Retrieve top-k from memory bank
                x = self.encoder(latents, x) # (B, L, D)
                # retrievals = self.retrieve(x).view(batch_size, -1, x.shape[-1]) # (B, K*L, D)
                retrievals = self.retrieve(x, attn_output_weights=None, is_cos_sim=is_cos_sim).view(batch_size, -1, x.shape[-1]) # (B, K*L, D)

            # (B, K, 2L, D)
            # Average over K retrievals

            x = torch.cat([x, retrievals], dim=1) # (B, (K+1)*L, D)
            x = self.blocks(x)
            x = self.decoder(query, x).reshape(batch_size, -1)
            x = self.proj(x)
        else:
            # ensemble
            batch_size = len(tf)
            latents = self.latents.repeat(batch_size, 1, 1)
            query = self.query.repeat(batch_size * self.top_k, 1, 1) # (B*K, 1, D)
            
            x, _ = self.tensor_frame_encoder(tf)
            x = x + self.pos_embedding
            x = self.encoder(latents, x)
            
            retrievals = self.retrieve(x) # (B, K, L, D)

            x = x.unsqueeze(1).repeat(1, self.top_k, 1, 1) # (B, K, L, D)
            x = torch.cat([x, retrievals], dim=2) # (B, K, 2L, D)
            x = x.view(batch_size * self.top_k, -1, self.hidden_dim) # (B*K, 2L, D)

            x = self.blocks(x) # (B*K, 2L, D)
            x = self.decoder(query, x) # (B*K, 1, D)
            x = self.proj(x) #  (B*K, 1, num_classes)
            x = x.view(batch_size, self.top_k, -1).mean(dim=1) 
            
        return x