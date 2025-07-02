from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn as nn

import torch_frame
from torch_frame import TensorFrame, stype
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
    StypeEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from encoder.piecewise_linear_encoder import StypeWiseFeatureEncoderCustom, PiecewiseLinearEncoder 


class TabM(nn.Module):
    r"""The light-weight MLP model that concats column embeddings and
    applies efficient ensemble of MLPs over it.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        out_channels: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder]
        | None = None,
        normalization: str | None = "layer_norm",
        dropout_prob: float = 0.2,
        bins_list: list[list[float]] | None = None,  
    ) -> None:
        super().__init__()

        if bins_list is not None:
            self.tensor_frame_encoder = StypeWiseFeatureEncoderCustom(
                out_channels=embed_dim,
                col_stats=col_stats,
                col_names_dict=col_names_dict,
                bins_list=bins_list, # pass pre-computed bins_list for numerical features.
                stype_encoder_dict={
                    stype.categorical: EmbeddingEncoder(),
                    stype.numerical: PiecewiseLinearEncoder(),
                }
            )
        else:
            # Linear Encoder for numerical features
            self.encoder = StypeWiseFeatureEncoder(
                out_channels=embed_dim,
                col_stats=col_stats,
                col_names_dict=col_names_dict,
                stype_encoder_dict={
                    stype.categorical: EmbeddingEncoder(),
                    stype.numerical: LinearEncoder(), # Linear Encoder for numerical features
                }
            )


        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def forward(self, tf: TensorFrame) -> Tensor:
        batch_size = len(tf)
        x, _ = self.encoder(tf) # (B, F, D') 
        x = x.view(batch_size, -1) # (B, F * D')
        x = self.

        x = torch.mean(x, dim=1)

        out = self.mlp(x)
        return out