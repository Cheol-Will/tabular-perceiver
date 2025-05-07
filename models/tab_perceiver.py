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
    _, _, _, head_dim = query.shape
    
    attention = F.softmax(torch.einsum("bhqd,bhkd->bhqk", query, key) / (head_dim**(0.5)), dim=-1) 
    attention = F.dropout(attention, p=dropout_prob, training=train)
    weighted_sum = torch.einsum("bhqk,bhkd->bhqd", attention, value) # (batch_size, num_heads, query_len, head_dim)
    return weighted_sum


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
        out = attend(Q, K, V, self.dropout_prob, self.training) 
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        out = self.proj(out)
        return out


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
        x = x + self.attention(self.norm1(x))
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

    def forward(self, query, key, expert_idx=None):
        x = query + self.attention(self.q_norm(query), self.kv_norm(key))
        x = x + self.mlp(self.mlp_norm(x))
        if expert_idx is not None:
            x = x + self.moe[expert_idx](self.mlp_norm(x))
        return x


class TabPerceiver(nn.Module):
    r"""

    Args:
        num_classes (int): Output channels dimensionality
        num_heads (int): Number of heads in the self-attention layer.
        num_layers (int): Number of self-attention layers
        num_latents (int): Number of latents
        hidden_dim (int): Embedding Dimensionality
        mlp_ratio (float): Multiplier in MLP
    """
    def __init__(
        self,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        num_latents: int,
        hidden_dim: int,
        mlp_ratio: float,
        dropout_prob: float,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
    ) -> None:
        super(TabPerceiver, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_features = self.calculate_num_features(col_names_dict)
        if stype_encoder_dict is None:
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
        }
        
        # (num_features, 1) -> (num_features, hidden_dim) 
        self.tensor_frame_encoder = StypeWiseFeatureEncoder(
            out_channels=hidden_dim,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        # Positional embedding 
        self.pos_embedding = nn.Parameter(torch.empty(1, self.num_features, hidden_dim))
        
        # Latents and Decoder query with shape of (1, N, D) and (1, 1, D)
        self.latents = nn.Parameter(torch.empty(1, num_latents, hidden_dim))
        self.queries = nn.Parameter(torch.empty(1, 1, hidden_dim)) 
        self.encoder = CrossAttention(
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
        )
        self.blocks = nn.Sequential(
            *[SelfAttention(
                hidden_dim=hidden_dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_prob=dropout_prob,
            )
            for _ in range(num_layers)]
        )
        self.decoder = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        self.reset_parameters()
        
    def calculate_num_features(self, col_names_dict):
        num_features = 0
        for k, v in col_names_dict.items():
            num_features += len(v)
        return num_features        

    def reset_parameters(self) -> None:
        # tensor_frame embedding parameter reset
        self.tensor_frame_encoder.reset_parameters()

        # truncated normal with std=0.02(default) from PerceiverIO 
        nn.init.normal_(self.pos_embedding)
        nn.init.trunc_normal_(self.latents, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()

        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()

    def forward(self, tf):
        # pre-processing with shape (batch_size, num_colummns, hidden_dim)
        batch_size = len(tf)
        x, _ = self.tensor_frame_encoder(tf)
        x = x + self.pos_embedding

        # Encode input into latent of shape (batch_size, num_latents, hidden_dim) 
        latents = self.latents.repeat(batch_size, 1, 1)
        x = self.encoder(latents, x)
        
        # Transformer Blocks
        x = self.blocks(x)

        # Decode and projection: (batch_size, hidden_dim) -> (batch_size, num_classes)
        queries = self.queries.repeat(batch_size, 1, 1)
        x = self.decoder(queries, x).reshape(batch_size, -1)
        x = self.proj(x)
        return x


class TabPerceiverMultiTask(nn.Module):
    def __init__(
        self,
        num_classes: list,
        num_heads: int,
        num_layers: int,
        num_latents: int,
        hidden_dim: int,
        mlp_ratio: float,
        dropout_prob: float,
        col_stats: list,
        col_names_dicts: list,
        moe_ratio: float = None,
        is_moe: bool = False,
    ):
        super(TabPerceiverMultiTask, self).__init__()
        self.num_tasks = len(col_stats)
        self.is_moe = is_moe
        num_experts = self.num_tasks if is_moe else None
        num_features_list = self.calculate_num_features(col_names_dicts)
        
        self.tensor_frame_encoders = nn.ModuleList([
            StypeWiseFeatureEncoder(
                out_channels=hidden_dim,
                col_stats=col_stats[i],
                col_names_dict=col_names_dicts[i],
                stype_encoder_dict={
                    stype.categorical: EmbeddingEncoder(),
                    stype.numerical: LinearEncoder(),
                }
            )
            for i in range(self.num_tasks)
        ])
        self.pos_embeddings = nn.ParameterList([
            nn.Parameter(torch.empty(1, num_features_list[i], hidden_dim)) 
            for i in range(self.num_tasks)
        ])
        
        self.latents = nn.Parameter(torch.empty(1, num_latents, hidden_dim))
        self.queries = nn.ParameterList([
            nn.Parameter(torch.empty(1, 1, hidden_dim)) 
            for i in range(self.num_tasks)
        ])
        self.encoder = CrossAttention(
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
            num_experts=num_experts,
            moe_ratio=moe_ratio,
        )
        # self.blocks = nn.Sequential(
        #     *[SelfAttention(
        #         hidden_dim=hidden_dim, 
        #         num_heads=num_heads,
        #         mlp_ratio=4,
        #         dropout_prob=dropout_prob,
        #         num_experts=num_experts,
        #     )
        #     for _ in range(num_layers)]
        # )
        self.blocks = nn.ModuleList(
            [SelfAttention(
                hidden_dim=hidden_dim, 
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_prob=dropout_prob,
                num_experts=num_experts,
            moe_ratio=moe_ratio,
            )
            for _ in range(num_layers)]
        )
        self.decoder = CrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout_prob,
            num_experts=num_experts,
            moe_ratio=moe_ratio,
        )
        
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes[i])
            for i in range(self.num_tasks)
        ])
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # tensor_frame embedding parameter reset
        for tensor_frame_encoder in self.tensor_frame_encoders: 
            tensor_frame_encoder.reset_parameters()

        # truncated normal with std=0.02(default) from PerceiverIO 
        for pos_embedding in self.pos_embeddings:
            nn.init.normal_(pos_embedding)

        nn.init.trunc_normal_(self.latents, std=0.02)
        for query in self.queries:
            nn.init.trunc_normal_(query, std=0.02)

        self.encoder.reset_parameters()
        self.decoder.reset_parameters()

        for block in self.blocks:
            block.reset_parameters()

        for projection in self.projections:
            projection.reset_parameters()

    def calculate_num_features(self, col_names_dicts):
        num_features_list = []
        for col_names_dict in col_names_dicts:
            num_features = 0
            for k, v in col_names_dict.items():
                num_features += len(v)
            num_features_list.append(num_features)

        return num_features_list

    def forward(self, tf, task_idx):
        expert_idx = task_idx if self.is_moe else None

        batch_size = len(tf)
        x, _ = self.tensor_frame_encoders[task_idx](tf)
        x = x + self.pos_embeddings[task_idx]

        latents = self.latents.repeat(batch_size, 1, 1)
        x = self.encoder(latents, x, expert_idx)

        for block in self.blocks:
            x = block(x, expert_idx)
        # x = self.blocks(x, expert_idx)

        queries = self.queries[task_idx].repeat(batch_size, 1, 1)
        x = self.decoder(queries, x, expert_idx).reshape(batch_size, -1)
        x = self.projections[task_idx](x)
        return x


class TabPerceiverTransfer(TabPerceiver):
    def reconstructIO(
        self,
        num_classes: int,
        num_features: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
    ):
        stype_encoder_dict = {
            stype.categorical: EmbeddingEncoder(),
            stype.numerical: LinearEncoder(),
        }
        self.tensor_frame_encoder = StypeWiseFeatureEncoder(
            out_channels=self.hidden_dim,
            col_stats=col_stats,
            col_names_dict=col_names_dict,
            stype_encoder_dict=stype_encoder_dict,
        )
        self.pos_embedding = nn.Parameter(torch.empty(1, num_features, self.hidden_dim))
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, num_classes)
        )           
        self.freeze_transformer()
        self.reset_parameters_finetune()

    def freeze_transformer(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.latents.requires_grad = False
        self.queries.requires_grad = False
        
    def reset_parameters_finetune(self):
        self.tensor_frame_encoder.reset_parameters()
        torch.nn.init.normal_(self.pos_embedding)
        self.proj[0].reset_parameters()
        self.proj[1].reset_parameters()