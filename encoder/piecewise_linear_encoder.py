from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import ModuleDict

import torch_frame
from torch_frame import TensorFrame
from torch_frame.data.stats import StatType
from torch_frame.nn.encoder import FeatureEncoder
from torch_frame.nn.encoder.stype_encoder import StypeEncoder

import warnings
from typing import Any, List, Optional
from tqdm import tqdm

from torch.nn import (
    Parameter,
)

from torch_frame import NAStrategy, stype
from torch_frame.data.stats import StatType

try:
    import sklearn.tree as sklearn_tree
except ImportError:
    sklearn_tree = None



def _check_bins(bins: list[Tensor]) -> None:
    if not bins:
        raise [ValueError]('The list of bins must not be empty')
    for i, feature_bins in enumerate(bins):
        if not isinstance(feature_bins, Tensor):
            raise ValueError(
                'bins must be a list of PyTorch tensors. '
                f'However, for {i=}: {type(bins[i])=}'
            )
        if feature_bins.ndim != 1:
            raise ValueError(
                'Each item of the bin list must have exactly one dimension.'
                f' However, for {i=}: {bins[i].ndim=}'
            )
        if len(feature_bins) < 2:
            raise ValueError(
                'All features must have at least two bin edges.'
                f' However, for {i=}: {len(bins[i])=}'
            )
        if not feature_bins.isfinite().all():
            raise ValueError(
                'Bin edges must not contain nan/inf/-inf.'
                f' However, this is not true for the {i}-th feature'
            )
        if (feature_bins[:-1] >= feature_bins[1:]).any():
            raise ValueError(
                'Bin edges must be sorted.'
                f' However, the for the {i}-th feature, the bin edges are not sorted'
            )
        if len(feature_bins) == 2:
            warnings.warn(
                f'The {i}-th feature has just two bin edges, which means only one bin.'
                ' Strictly speaking, using a single bin for the'
                ' piecewise-linear encoding should not break anything,'
                ' but it is the same as using sklearn.preprocessing.MinMaxScaler'
            )


def compute_bins(
    X: torch.Tensor,
    n_bins: int = 48,
    *,
    tree_kwargs: Optional[dict[str, Any]] = None,
    y: Optional[Tensor] = None,
    regression: Optional[bool] = None,
    verbose: bool = False,
) -> list[Tensor]:
    """Compute the bin boundaries for `PiecewiseLinearEncoding` and `PiecewiseLinearEmbeddings`.

    **Usage**

    Compute bins using quantiles (Section 3.2.1 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> bins = compute_bins(X_train)

    Compute bins using decision trees (Section 3.2.2 in the paper):

    >>> X_train = torch.randn(10000, 2)
    >>> y_train = torch.randn(len(X_train))
    >>> bins = compute_bins(
    ...     X_train,
    ...     y=y_train,
    ...     regression=True,
    ...     tree_kwargs={'min_samples_leaf': 64, 'min_impurity_decrease': 1e-4},
    ... )

    Args:
        X: the training features.
        n_bins: the number of bins.
        tree_kwargs: keyword arguments for `sklearn.tree.DecisionTreeRegressor`
            (if ``regression=True``) or `sklearn.tree.DecisionTreeClassifier`
            (if ``regression=False``).
            NOTE: requires ``scikit-learn>=1.0,>2`` to be installed.
        y: the training labels (must be provided if ``tree`` is not None).
        regression: whether the labels are regression labels
            (must be provided if ``tree`` is not None).
        verbose: if True and ``tree_kwargs`` is not None, than ``tqdm``
            (must be installed) will report the progress while fitting trees.

    Returns:
        A list of bin edges for all features. For one feature:

        - the maximum possible number of bin edges is ``n_bins + 1``.
        - the minimum possible number of bin edges is ``1``.
    """  # noqa: E501
    if not isinstance(X, Tensor):
        raise ValueError(f'X must be a PyTorch tensor, however: {type(X)=}')
    if X.ndim != 2:
        raise ValueError(f'X must have exactly two dimensions, however: {X.ndim=}')
    if X.shape[0] < 2:
        raise ValueError(f'X must have at least two rows, however: {X.shape[0]=}')
    if X.shape[1] < 1:
        raise ValueError(f'X must have at least one column, however: {X.shape[1]=}')
    if not X.isfinite().all():
        raise ValueError('X must not contain nan/inf/-inf.')
    if (X == X[0]).all(dim=0).any():
        raise ValueError(
            'All columns of X must have at least two distinct values.'
            ' However, X contains columns with just one distinct value.'
        )
    if n_bins <= 1 or n_bins >= len(X):
        raise ValueError(
            'n_bins must be more than 1, but less than len(X), however:'
            f' {n_bins=}, {len(X)=}'
        )

    if tree_kwargs is None:
        if y is not None or regression is not None or verbose:
            raise ValueError(
                'If tree_kwargs is None, then y must be None, regression must be None'
                ' and verbose must be False'
            )

        _upper = 2**24  # 16_777_216
        if len(X) > _upper:
            warnings.warn(
                f'Computing quantile-based bins for more than {_upper} million objects'
                ' may not be possible due to the limitation of PyTorch'
                ' (for details, see https://github.com/pytorch/pytorch/issues/64947;'
                ' if that issue is successfully resolved, this warning may be irrelevant).'  # noqa
                ' As a workaround, subsample the data, i.e. instead of'
                '\ncompute_bins(X, ...)'
                '\ndo'
                '\ncompute_bins(X[torch.randperm(len(X), device=X.device)[:16_777_216]], ...)'  # noqa
                '\nOn CUDA, the computation can still fail with OOM even after'
                ' subsampling. If this is the case, try passing features by groups:'
                '\nbins = sum('
                '\n    compute_bins(X[:, idx], ...)'
                '\n    for idx in torch.arange(len(X), device=X.device).split(group_size),'  # noqa
                '\n    start=[]'
                '\n)'
                '\nAnother option is to perform the computation on CPU:'
                '\ncompute_bins(X.cpu(), ...)'
            )
        del _upper
        bins = [
            q.unique()
            for q in torch.quantile(
                X, torch.linspace(0.0, 1.0, n_bins + 1).to(X), dim=0
            ).T
        ]
        _check_bins(bins)
        return bins

    else:
        if sklearn_tree is None:
            raise RuntimeError(
                'The scikit-learn package is missing.'
                ' See README.md for installation instructions'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if y.ndim != 1:
            raise ValueError(f'y must have exactly one dimension, however: {y.ndim=}')
        if len(y) != len(X):
            raise ValueError(
                f'len(y) must be equal to len(X), however: {len(y)=}, {len(X)=}'
            )
        if y is None or regression is None:
            raise ValueError(
                'If tree_kwargs is not None, then y and regression must not be None'
            )
        if 'max_leaf_nodes' in tree_kwargs:
            raise ValueError(
                'tree_kwargs must not contain the key "max_leaf_nodes"'
                ' (it will be set to n_bins automatically).'
            )

        if verbose:
            if tqdm is None:
                raise ImportError('If verbose is True, tqdm must be installed')
            tqdm_ = tqdm
        else:
            tqdm_ = lambda x: x  # noqa: E731

        if X.device.type != 'cpu' or y.device.type != 'cpu':
            warnings.warn(
                'Computing tree-based bins involves the conversion of the input PyTorch'
                ' tensors to NumPy arrays. The provided PyTorch tensors are not'
                ' located on CPU, so the conversion has some overhead.',
                UserWarning,
            )
        X_numpy = X.cpu().numpy()
        y_numpy = y.cpu().numpy()
        bins = []
        for column in tqdm_(X_numpy.T):
            feature_bin_edges = [float(column.min()), float(column.max())]
            tree = (
                (
                    sklearn_tree.DecisionTreeRegressor
                    if regression
                    else sklearn_tree.DecisionTreeClassifier
                )(max_leaf_nodes=n_bins, **tree_kwargs)
                .fit(column.reshape(-1, 1), y_numpy)
                .tree_
            )
            for node_id in range(tree.node_count):
                # The following condition is True only for split nodes. Source:
                # https://scikit-learn.org/1.0/auto_examples/tree/plot_unveil_tree_structure.html#tree-structure
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    feature_bin_edges.append(float(tree.threshold[node_id]))
            bins.append(torch.as_tensor(feature_bin_edges).unique())
        _check_bins(bins)
        return [x.to(device=X.device, dtype=X.dtype) for x in bins]


# _NLinear is a simplified copy of delu.nn.NLinear:
# https://yura52.github.io/delu/stable/api/generated/delu.nn.NLinear.html
class _NLinear(nn.Module):
    """N *separate* linear layers for N feature embeddings.

    In other words,
    each feature embedding is transformed by its own dedicated linear layer.
    """

    def __init__(
        self, n: int, in_features: int, out_features: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(torch.empty(n, in_features, out_features))
        self.bias = Parameter(torch.empty(n, out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters."""
        d_in_rsqrt = self.weight.shape[-2] ** -0.5
        nn.init.uniform_(self.weight, -d_in_rsqrt, d_in_rsqrt)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -d_in_rsqrt, d_in_rsqrt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        if x.ndim != 3:
            raise ValueError(
                '_NLinear supports only inputs with exactly one batch dimension,'
                ' so `x` must have a shape like (BATCH_SIZE, N_FEATURES, D_EMBEDDING).'
            )
        assert x.shape[-(self.weight.ndim - 1) :] == self.weight.shape[:-1]

        x = x.transpose(0, 1)
        x = x @ self.weight
        x = x.transpose(0, 1)
        if self.bias is not None:
            x = x + self.bias
        return x


class _PiecewiseLinearEncodingImpl(nn.Module):
    """Piecewise-linear encoding.

    NOTE: THIS CLASS SHOULD NOT BE USED DIRECTLY.
    In particular, this class does *not* add any positional information
    to feature encodings. Thus, for Transformer-like models,
    `PiecewiseLinearEmbeddings` is the only valid option.

    Note:
        This is the *encoding* module, not the *embedding* module,
        so it only implements Equation 1 (Figure 1) from the paper,
        and does not have trainable parameters.

    **Shape**

    * Input: ``(*, n_features)``
    * Output: ``(*, n_features, max_n_bins)``,
      where ``max_n_bins`` is the maximum number of bins over all features:
      ``max_n_bins = max(len(b) - 1 for b in bins)``.

    To understand the output structure,
    consider a feature with the number of bins ``n_bins``.
    Formally, its piecewise-linear encoding is a vector of the size ``n_bins``
    that looks as follows::

        x_ple = [1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]

    However, this class will instead produce a vector of the size ``max_n_bins``::

        x_ple_actual = [*x_ple[:-1], *zeros(max_n_bins - n_bins), x_ple[-1]]

    In other words:

    * The last encoding component is **always** located in the end,
      even if ``n_bins == 1`` (i.e. even if it is the only component).
    * The leading ``n_bins - 1`` components are located in the beginning.
    * Everything in-between is always set to zeros (like "padding", but in the middle).

    This implementation is *significantly* faster than the original one.
    It relies on two key observations:

    * The piecewise-linear encoding is just
      a non-trainable linear transformation followed by a clamp-based activation.
      Pseudocode: `PiecewiseLinearEncoding(x) = Activation(Linear(x))`.
      The parameters of the linear transformation are defined by the bin edges.
    * Aligning the *last* encoding channel across all features
      allows applying the aforementioned activation simultaneously to all features
      without the loop over features.
    """

    weight: Tensor
    """The weight of the linear transformation mentioned in the class docstring."""

    bias: Tensor
    """The bias of the linear transformation mentioned in the class docstring."""

    single_bin_mask: Optional[Tensor]
    """The indicators of the features with only one bin."""

    mask: Optional[Tensor]
    """The indicators of the "valid" (i.e. "non-padding") part of the encoding."""

    def __init__(self, bins: list[Tensor]) -> None:
        """
        Args:
            bins: the bins computed by `compute_bins`.
        """
        assert len(bins) > 0
        super().__init__()

        n_features = len(bins)
        n_bins = [len(x) - 1 for x in bins]
        max_n_bins = max(n_bins)

        self.register_buffer('weight', torch.zeros(n_features, max_n_bins))
        self.register_buffer('bias', torch.zeros(n_features, max_n_bins))

        single_bin_mask = torch.tensor(n_bins) == 1
        self.register_buffer(
            'single_bin_mask', single_bin_mask if single_bin_mask.any() else None
        )

        self.register_buffer(
            'mask',
            # The mask is needed if features have different number of bins.
            None
            if all(len(x) == len(bins[0]) for x in bins)
            else torch.row_stack(
                [
                    torch.cat(
                        [
                            # The number of bins for this feature, minus 1:
                            torch.ones((len(x) - 1) - 1, dtype=torch.bool),
                            # Unused components (always zeros):
                            torch.zeros(max_n_bins - (len(x) - 1), dtype=torch.bool),
                            # The last bin:
                            torch.ones(1, dtype=torch.bool),
                        ]
                    )
                    # x is a tensor containing the bin bounds for a given feature.
                    for x in bins
                ]
            ),
        )

        for i, bin_edges in enumerate(bins):
            # Formally, the piecewise-linear encoding of one feature looks as follows:
            # `[1, ..., 1, (x - this_bin_left_edge) / this_bin_width, 0, ..., 0]`
            # The linear transformation based on the weight and bias defined below
            # implements the expression in the middle before the clipping to [0, 1].
            # Note that the actual encoding layout produced by this class
            # is slightly different. See the docstring of this class for details.
            bin_width = bin_edges.diff()
            w = 1.0 / bin_width
            b = -bin_edges[:-1] / bin_width
            # The last encoding component:
            self.weight[i, -1] = w[-1]
            self.bias[i, -1] = b[-1]
            # The leading encoding components:
            self.weight[i, : n_bins[i] - 1] = w[:-1]
            self.bias[i, : n_bins[i] - 1] = b[:-1]
            # All in-between components will always be zeros,
            # because the weight and bias are initialized with zeros.

    def get_max_n_bins(self) -> int:
        return self.weight.shape[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Do the forward pass."""
        x = torch.addcmul(self.bias, self.weight, x[..., None])
        if x.shape[-1] > 1:
            x = torch.cat(
                [
                    x[..., :1].clamp_max(1.0),
                    x[..., 1:-1].clamp(0.0, 1.0),
                    (
                        x[..., -1:].clamp_min(0.0)
                        if self.single_bin_mask is None
                        else torch.where(
                            # For features with only one bin,
                            # the whole "piecewise-linear" encoding effectively behaves
                            # like mix-max scaling
                            # (assuming that the edges of the single bin
                            #  are the minimum and maximum feature values).
                            self.single_bin_mask[..., None],
                            x[..., -1:],
                            x[..., -1:].clamp_min(0.0),
                        )
                    ),
                ],
                dim=-1,
            )
        return x


class PiecewiseLinearEncoder(StypeEncoder):
    r"""A numerical converter that transforms a tensor into a piecewise
    linear representation with learnable embeddings per feature.
    Fetched from yandex-research's rtdl_num_embeddings.
    """
    LAZY_ATTRS = StypeEncoder.LAZY_ATTRS | {"bins_list"}    # init waits until bins_list is assgined
    supported_stypes = {stype.numerical}

    def __init__(
        self,
        out_channels: int | None = None,
        stats_list: Optional[List[dict[StatType, Any]]] = None,
        stype: Optional[stype] = None,
        post_module: Optional[nn.Module] = None,
        na_strategy: Optional[NAStrategy] = None,
        bins_list: Optional[list[list[float]]] = None,
    ) -> None:
        super().__init__(out_channels, stats_list, stype, post_module, na_strategy)
        self.bins_list = bins_list

    def init_modules(self) -> None:
        super().init_modules()
        # Convert lists of floats to list of 1D tensors
        bins: List[Tensor] = [torch.as_tensor(b, dtype=torch.float32) for b in self.bins_list]
        # _check_bins(bins)

        # Piecewise-linear encoding implementation from rtdl_num_embeddings (no trainable params)
        self.impl = _PiecewiseLinearEncodingImpl(bins)

        # per-feature linear embeddings
        self.linear = _NLinear(
            len(bins),
            self.impl.get_max_n_bins(),
            self.out_channels,
            bias=True,
        )
        # Initialize embedding weights
        nn.init.normal_(self.linear.weight, std=0.01)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    def reset_parameters(self) -> None:
        # This class does not have its own parameters.
        # class _Linear calls its own reset_parameters 
        pass


    def encode_forward(
        self,
        feat: Tensor,
        col_names: Optional[List[str]] = None,
    ) -> Tensor:
        # input: (B, F)
        x_ple = self.impl(feat) # (B, F, max_num_bins), note that max_num_bins is the largest num_bins among features. 
        # (B, F, D)
        x_emb = self.linear(x_ple)
        return x_emb


class StypeWiseFeatureEncoderCustom(FeatureEncoder):
    r"""Feature encoder that transforms each stype tensor into embeddings and
    performs the final concatenation.
        Note that this custom encoder is for target-aware PLE.
    Args:
        out_channels (int): Output dimensionality.
        col_stats
            (dict[str, dict[:class:`torch_frame.data.stats.StatType`, Any]]):
            A dictionary that maps column name into stats. Available as
            :obj:`dataset.col_stats`.
        col_names_dict (dict[:class:`torch_frame.stype`, list[str]]): A
            dictionary that maps stype to a list of column names. The column
            names are sorted based on the ordering that appear in
            :obj:`tensor_frame.feat_dict`.
            Available as :obj:`tensor_frame.col_names_dict`.
        stype_encoder_dict
            (dict[:class:`torch_frame.stype`,
            :class:`torch_frame.nn.encoder.StypeEncoder`]):
            A dictionary that maps :class:`torch_frame.stype` into
            :class:`torch_frame.nn.encoder.StypeEncoder` class. Only
            parent :class:`stypes <torch_frame.stype>` are supported
            as keys.
    """
    def __init__(
        self,
        out_channels: int,
        col_stats: dict[str, dict[StatType, Any]],
        col_names_dict: dict[torch_frame.stype, list[str]],
        stype_encoder_dict: dict[torch_frame.stype, StypeEncoder],
        bins_list: list[list[float]] | None = None,  

    ) -> None:
        super().__init__()

        self.col_stats = col_stats
        self.col_names_dict = col_names_dict
        self.encoder_dict = ModuleDict()
        for stype, stype_encoder in stype_encoder_dict.items():
            if stype != stype.parent:
                if stype.parent in stype_encoder_dict:
                    msg = (
                        f"You can delete this {stype} directly since encoder "
                        f"for parent stype {stype.parent} is already declared."
                    )
                else:
                    msg = (f"To resolve the issue, you can change the key from"
                           f" {stype} to {stype.parent}.")
                raise ValueError(f"{stype} is an invalid stype to use in the "
                                 f"stype_encoder_dcit. {msg}")
            if stype not in stype_encoder.supported_stypes:
                raise ValueError(
                    f"{stype_encoder} does not support encoding {stype}.")

            if stype in col_names_dict:
                stats_list = [
                    self.col_stats[col_name]
                    for col_name in self.col_names_dict[stype]
                ]
                # Set lazy attributes
                stype_encoder.stype = stype
                stype_encoder.out_channels = out_channels
                stype_encoder.stats_list = stats_list
                self.encoder_dict[stype.value] = stype_encoder
                if stype == torch_frame.stype.numerical:
                    stype_encoder.bins_list = bins_list

    def forward(self, tf: TensorFrame) -> tuple[Tensor, list[str]]:
        all_col_names = []
        xs = []
        for stype in tf.stypes:
            feat = tf.feat_dict[stype]
            col_names = self.col_names_dict[stype]
            x = self.encoder_dict[stype.value](feat, col_names)
            xs.append(x)
            all_col_names.extend(col_names)
        x = torch.cat(xs, dim=1)
        return x, all_col_names
