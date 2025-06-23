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
        if len(query.shape) == 3:
            B, N, D = query.shape
            H = self.num_heads
            head_dim = self.head_dim
            Q = self._query(query)
            K = self._key(key)
            V = self._value(value)
            Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
            K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        elif len(query.shape) == 4:
            # row, col-wise 
            B, N, F, D = query.shape
            q_flat = query.view(B * N, F, D)
            k_flat = key.view(B * N, F, D)
            v_flat = value.view(B * N, F, D)
            Q = self._query(q_flat).view(B * N, F, self.num_heads, self.head_dim).transpose(1, 2) # (BN, H, F, head_dim)
            K = self._key(k_flat).view(B * N, F, self.num_heads, self.head_dim).transpose(1, 2)
            V = self._value(v_flat).view(B * N, F, self.num_heads, self.head_dim).transpose(1, 2)

        else:
            raise ValueError(f"length of query shape must be 3 or 4, but got {query.shape}")    

        attn_output, attn_output_weights = attend(Q, K, V, self.dropout_prob, self.training) # output shape == query shape
        if len(query.shape) == 3:
            out = attn_output.permute(0, 2, 1, 3).reshape(B, N, -1) # (B, N, D)
        elif len(query.shape) == 4:
            out = attn_output.permute(0, 2, 1, 3).reshape(B * N, F, -1) # (BN, F, D)
        out = self.proj(out) # (B, N, D) or (BN, F, D)

        if len(query.shape) == 4:
            out = out.view(B, N, F, D)

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

class RowColAttention(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            mlp_ratio,
            dropout_prob, 
    ):
        super(RowColAttention, self).__init__()
        self.col_wise_attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,

        )
        self.row_wise_attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
        )

    def reset_parameters(self):
        self.col_wise_attention.reset_parameters()
        self.row_wise_attention.reset_parameters()

    def forward(self, x):
        # input: (B, K+1, F, D) -> output: (B, K+1, F, D) 
        x = self.col_wise_attention(x)
        x = x.permute(0, 2, 1, 3)
        x = self.row_wise_attention(x)
        x = x.permute(0, 2, 1, 3) 
        return x

class MemGlovalAvgPool(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_samples: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        mlp_ratio: float,
        dropout_prob: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        top_k: int = 5,
    ):
        super(MemGlovalAvgPool, self).__init__()

        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.num_features = self.calculate_num_features(col_names_dict)
        
        self.register_buffer('memory', torch.empty(num_samples, self.num_features, hidden_dim))
        self.row_pos_embedding = nn.Parameter(torch.empty(1, self.top_k + 1, 1, hidden_dim)) # (1, k+1, 1, D)
        self.col_pos_embedding = nn.Parameter(torch.empty(1, 1, self.num_features, hidden_dim)) # (1, 1, F, D)

        self.tensor_frame_encoder = StypeWiseFeatureEncoder(
            out_channels=hidden_dim,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict={
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }
        )
        self.blocks = nn.Sequential(
            *[
                RowColAttention(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout_prob=dropout_prob,
                ) 
                for _ in range(num_layers)
            ]
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
        nn.init.normal_(self.row_pos_embedding)
        nn.init.normal_(self.col_pos_embedding)
        self.tensor_frame_encoder.reset_parameters()
        
        for block in self.blocks:
            block.reset_parameters()

        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()

    @torch.no_grad()
    def retrieve(self, x, indicies = None, chunk_size = 1024):
        r"""
            Retrieve top k instances from memory bank.
            input: 
                x: Encoded input (batch_size, num_features, hidden_dim)
                indicies: Index of input during training (batch_size)
                    so that model can excldue current target sample. 
            return: 
                tensor (batch_size, k, num_features, hidden_dim)
        """
        B, F, D = x.shape
        N = self.memory.size(0)
        K = min(self.top_k, N)

        x_flat   = x.view(B, -1)                      # (B, F*D)
        mem_flat = self.memory.view(N, -1)            # (N, F*D)
        x_squared = (x_flat**2).sum(dim=1, keepdim=True)     # (B, 1)

        dist_avg = torch.empty(B, N, device=x.device)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N) 
            mem = mem_flat[start:end]           # (C, F*D) 
            mem_squared = (mem**2).sum(dim=1)    # (C,)
            
            # inner product: (B, F*D) x (F*D, C) -> (B, C)
            cross_term = x_flat.matmul(mem.t())     # (B, C)
            dist = x_squared + mem_squared.unsqueeze(0) - 2 * cross_term
            dist_avg[:, start:end] = dist / float(F)  

        # exclude target sample
        if self.training and indicies is not None:
            idx = torch.arange(B, device=x.device)
            dist_avg[idx, indicies] = float("inf")

        _, topk = torch.topk(-dist_avg, k=K, dim=-1)      # (B, K)
        retrievals = self.memory[topk.view(-1)]           # (B*K, F, D)
        return retrievals.view(B, K, F, D)                # (B, K, F, D)

    @torch.no_grad()
    def update_memory(self, dataloader):
        self.eval()
        all_indices = []
        all_representations = []

        for tf, index in dataloader:
            tf = tf.to(self.memory.device)
            index = index.to(self.memory.device)
            x, _ = self.tensor_frame_encoder(tf)       
            
            all_indices.append(index)
            all_representations.append(x.detach())     

        idx = torch.cat(all_indices, dim=0)            
        feat = torch.cat(all_representations, dim=0)   
        self.memory.index_copy_(0, idx, feat)          

    def forward(self, tf, indicies = None):
        if self.training and indicies is None:
            raise ValueError("During trainig, indicies should be provided.")

        batch_size = len(tf)
        col_pos_embedding = self.col_pos_embedding.repeat(batch_size, self.top_k + 1, 1, 1) # (1, 1, F, D) -> (B, K+1, F, D)
        row_pos_embedding = self.row_pos_embedding.repeat(batch_size, 1, self.num_features, 1) # (1, K+1, 1, D) -> (B, K+1, F, D)
        
        # feature embeddings and add col-wise positional embedding
        x, _ = self.tensor_frame_encoder(tf) # x, all_col_names
        
        # Retrieve top-k from memory bank and concat
        if self.training and indicies is None:
            # During training, exclude current input.
            retrievals = self.retrieve(x, indicies) # (B, K, F, D)
        else:
            retrievals = self.retrieve(x)
        x = x.unsqueeze(1) # (B, 1, F, D)
        x = torch.cat([x, retrievals], dim=1) # (B, K+1, F, D)

        # add col-wise positional embedding and row-wise positional embeddidng
        x = x + col_pos_embedding # (B, K+1, F, D) 
        x = x + row_pos_embedding # (B, K+1, F, D)
                    
        x = self.blocks(x) # (B, K+1, F, D)
        x = x.mean(dim=2) # (B, K+1, D)

        x = self.proj(x[:, 0, :])
        
        return x