import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
from functools import partial
from typing import Sequence
import math

class MonarchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 in_dims: Sequence[int], out_dims: Sequence[int],
                 bias: bool = True, checkpoint: bool = False,
                 ):
        super().__init__()
        print(np.prod(in_dims))
        print(in_features)
        print(np.prod(out_dims))
        print(out_features)
        assert len(in_dims) == len(out_dims) and len(in_dims) > 1
        assert np.prod(in_dims) == in_features
        assert np.prod(out_dims) == out_features
        self.in_features, self.out_features = in_features, out_features
        self.in_dims, self.out_dims = in_dims, out_dims
        self.checkpoint = checkpoint

        # construct weight tensors by keeping track of intermediate tensor dimension at each step
        self.weights = nn.ParameterList()
        current_numel = np.prod(in_dims)
        assert current_numel == in_features
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            self.weights.append(nn.Parameter(torch.empty(current_numel // in_dim, in_dim, out_dim)))
            current_numel = current_numel // in_dim * out_dim
        assert current_numel == out_features
        self.register_parameter('bias', nn.Parameter(torch.empty(out_features)) if bias else None)
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.0):
        # initialize, re-scale to account for the number of multiplied tensors
        init_std = (gain / np.sqrt(self.in_features)) ** (1 / len(self.in_dims))
        for weight in self.weights:
            nn.init.normal_(weight, std=init_std)
        if self.bias is not None:
            bound = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, _inside_checkpoint: bool = False):
        if self.checkpoint and not _inside_checkpoint and torch.is_grad_enabled():
            return checkpoint(partial(self.forward, _inside_checkpoint=True),
                              input if input.requires_grad else input.detach().requires_grad_(True),
                              preserve_rng_state=False)
        input_shape = input.shape
        tensor = input.view(-1, *self.in_dims)
        # shape: [flat_batch_size, in_dim[0], ..., in_dim[N]]

        del input
        tensor = tensor.permute(*np.roll(range(len(self.in_dims) + 1), -2))
        # new shape: [in_dim[1], ..., in_dim[N - 1], flat_batch_size, in_dim[0]]

        for i in range(len(self.weights)):
            # loop maintains tensor in the following shape: [*all_dims_except_i, batch, dim[i]]

            tensor = torch.bmm(
                tensor.flatten(0, -3), self.weights[i]
            ).view(*tensor.shape[:-1], -1)
            # ^-- BMM, output: [*other_dims, batch, out_dim[i]]
            #     left input:  [*other_dims, batch, in_dim[i]]
            #     right_input: [*other_dims, in_dim[i], out_dim[i]]

            # prepare next step, from [*other_dims, batch, out_dim[i]] to [*other_dims, batch, in_dim[i + 1]]
            tensor = tensor.swapaxes_(-1, i)
            # note: we can swap in-place because bmm does not need outputs for backprop

        # after loop: [out_dim[0], ..., out_dim[N - 1], batch]
        tensor = tensor.flatten(0, -2).swapaxes_(0, 1)
        tensor = tensor.reshape(*input_shape[:-1], -1)
        if self.bias is not None:
            tensor += self.bias
        return tensor
