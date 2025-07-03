from __future__ import annotations

from typing import Any, Literal

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

def init_rsqrt_uniform_(x: Tensor, d: int) -> Tensor:
    assert d > 0
    d_rsqrt = d**-0.5
    return nn.init.uniform_(x, -d_rsqrt, d_rsqrt)


@torch.inference_mode()
def init_random_signs_(x: Tensor) -> Tensor:
    return x.bernoulli_(0.5).mul_(2).add_(-1)

class LinearEfficientEnsemble(nn.Module):

    r: None | Tensor
    s: None | Tensor
    bias: None | Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        k: int,
        ensemble_scaling_in: bool,
        ensemble_scaling_out: bool,
        ensemble_bias: bool,
        scaling_init: Literal['ones', 'random-signs'],
    ):
        assert k > 0
        if ensemble_bias:
            assert bias
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_parameter(
            'r',
            (
                nn.Parameter(torch.empty(k, in_features))
                if ensemble_scaling_in
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            's',
            (
                nn.Parameter(torch.empty(k, out_features))
                if ensemble_scaling_out
                else None
            ),  # type: ignore[code]
        )
        self.register_parameter(
            'bias',
            (
                nn.Parameter(torch.empty(out_features))  # type: ignore[code]
                if bias and not ensemble_bias
                else nn.Parameter(torch.empty(k, out_features))
                if ensemble_bias
                else None
            ),
        )

        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.scaling_init = scaling_init

        self.reset_parameters()

    def reset_parameters(self):
        init_rsqrt_uniform_(self.weight, self.in_features)
        scaling_init_fn = {'ones': nn.init.ones_, 'random-signs': init_random_signs_}[
            self.scaling_init
        ]
        if self.r is not None:
            scaling_init_fn(self.r)
        if self.s is not None:
            scaling_init_fn(self.s)
        if self.bias is not None:
            bias_init = torch.empty(
                # NOTE: the shape of bias_init is (out_features,) not (k, out_features).
                # It means that all biases have the same initialization.
                # This is similar to having one shared bias plus
                # k zero-initialized non-shared biases.
                self.out_features,
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
            bias_init = init_rsqrt_uniform_(bias_init, self.in_features)
            with torch.inference_mode():
                self.bias.copy_(bias_init)

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, K, D)
        assert x.ndim == 3

        # >>> The equation (5) from the BatchEnsemble paper (arXiv v2).
        if self.r is not None:
            x = x * self.r
        x = x @ self.weight.T
        if self.s is not None:
            x = x * self.s
        # <<<

        if self.bias is not None:
            x = x + self.bias
        return x


class NLinear(nn.Module):
    """A stack of N linear layers. Each layer is applied to its own part of the input.

    **Shape**

    - Input: ``(B, N, in_features)``
    - Output: ``(B, N, out_features)``

    The i-th linear layer is applied to the i-th matrix of the shape (B, in_features).

    Technically, this is a simplified version of delu.nn.NLinear:
    https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html.
    The difference is that this layer supports only 3D inputs
    with exactly one batch dimension. By contrast, delu.nn.NLinear supports
    any number of batch dimensions.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        d = self.weight.shape[-2]
        init_rsqrt_uniform_(self.weight, d)
        if self.bias is not None:
            init_rsqrt_uniform_(self.bias, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x

    
class TabM(nn.Module):
    r"""TabM hat concats columns embeddings and apply batch ensemble. 
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        bins_list: list[list[float]] | None = None, 
        k: int = 32, 
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim # dimension for column embedding
        self.hidden_dim = hidden_dim
        self.num_features = self.calculate_num_features(col_names_dict)
        self.flat_embed_dim = self.num_features * embed_dim 
        self.k = k

        if bins_list is not None:
            self.encoder = StypeWiseFeatureEncoderCustom(
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
        # 
        self.backbone = nn.Sequential(*[
            LinearEfficientEnsemble(
                in_features=self.flat_embed_dim if i == 0 else hidden_dim,
                out_features=hidden_dim,
                bias=True,
                k=self.k,
                ensemble_scaling_in=True,
                ensemble_scaling_out=True,
                ensemble_bias=True,
                scaling_init='ones' # initialize scale parameters (adapter) with 1.
            )
            for i in range(num_layers)
        ])
        self.output = NLinear(k, hidden_dim, num_classes)
        self.reset_parameters()

    def calculate_num_features(self, col_names_dict):
        num_features = 0
        for k, v in col_names_dict.items():
            num_features += len(v)
        return num_features           

    def reset_parameters(self) -> None:
        # only for implementing abstract method. 
        # each module calls its own reset_parameters automatircally.
        pass

    def forward(self, tf: TensorFrame) -> Tensor:
        batch_size = len(tf)
        x, _ = self.encoder(tf) # (B, F, D') 
        x = x.view(batch_size, -1) # (B, F * D')
        x = x.unsqueeze(1).expand(-1, self.k, -1) # (B, K, F * D')

        x = self.backbone(x) # (B, K, D)
        x = self.output(x) # (B, K, C)

        if self.training:
            return x # (B, K, C)
        else:
            return x.mean(dim=1) # (B, C)